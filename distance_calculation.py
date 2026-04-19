import cv2
import time
import serial
from ultralytics import YOLO

# --- CONFIGURATION ---
CONF_LEVEL = 0.3 
UPDATE_INTERVAL = 0.1  # 100 ms interval for checking movement/sending commands
JITTER_THRESHOLD = 2.0 # Minimum pixel change required to trigger a status update
SMOOTHING_FACTOR = 0.4 # Between 0 and 1. Lower = smoother but slightly delayed

# --- SERIAL & MOVEMENT CONFIGURATION ---
SERIAL_PORT = "COM3"   # Change to your port (e.g., '/dev/ttyUSB0' on Linux/Mac, 'COM3' on Windows)
BAUD_RATE = 9600c
CENTER_DEADZONE = 0.15 # 15% of the frame width acts as a deadzone to prevent left/right jitter
SMALL_THRESHOLD = 0.3  # If the object's width is less than 30% of the frame width, it is considered "small"

# Initialize Serial Communication
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to serial port: {SERIAL_PORT}")
except Exception as e:
    print(f"Warning: Could not open serial port {SERIAL_PORT}. Error: {e}")
    ser = None

model = YOLO("best_ncnn_model") 
cap = cv2.VideoCapture(0)

# object_memory[id] stores: {"last_time": float, "last_width": float, "smoothed_width": float, "status": str}
object_memory = {}

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame_height, frame_width = frame.shape[:2]
    frame_center_x = frame_width / 2
    deadzone_pixels = frame_width * CENTER_DEADZONE

    current_time = time.time()
    results = model.track(frame, persist=True, conf=CONF_LEVEL, verbose=False)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().numpy()
        
        # To avoid sending conflicting serial commands for multiple objects,
        # we will only track and send commands based on the FIRST object detected in the list.
        primary_object_processed = False 

        for i, obj_id in enumerate(ids):
            x1, y1, x2, y2 = boxes[i]
            pixel_width = x2 - x1
            obj_center_x = (x1 + x2) / 2
            
            if pixel_width <= 0: continue
            obj_id = int(obj_id)

            # --- INITIALIZE NEW OBJECT ---
            if obj_id not in object_memory:
                object_memory[obj_id] = {
                    "last_time": current_time,
                    "last_width": pixel_width,
                    "smoothed_width": pixel_width,
                    "status": "Detecting..."
                }
            
            obj_data = object_memory[obj_id]
            
            # --- SMOOTH THE WIDTH TO PREVENT BOUNDING BOX JITTER ---
            obj_data["smoothed_width"] = (SMOOTHING_FACTOR * pixel_width) + ((1 - SMOOTHING_FACTOR) * obj_data["smoothed_width"])

            # --- PROCESS EVERY 100ms ---
            time_elapsed = current_time - obj_data["last_time"]
            
            if time_elapsed >= UPDATE_INTERVAL:
                # 1. Evaluate Z-Axis Depth (Original Logic)
                width_diff = obj_data["smoothed_width"] - obj_data["last_width"]
                
                if width_diff > JITTER_THRESHOLD:
                    obj_data["status"] = "Moving Closer"
                elif width_diff < -JITTER_THRESHOLD:
                    obj_data["status"] = "Moving Away"
                else:
                    obj_data["status"] = "Stationary / Lateral"
                
                # 2. Evaluate X-Axis Position & Size for Serial Commands
                if not primary_object_processed and ser is not None:
                    serial_cmd = None
                    
                    # Check if left of the deadzone
                    if obj_center_x < (frame_center_x - deadzone_pixels):
                        serial_cmd = "rotateleft\n"
                    # Check if right of the deadzone
                    elif obj_center_x > (frame_center_x + deadzone_pixels):
                        serial_cmd = "rotateright\n"
                    # If centered, check if it's small enough to move forward
                    elif pixel_width < (frame_width * SMALL_THRESHOLD):
                        serial_cmd = "forward\n"
                    
                    # Send command over serial
                    if serial_cmd:
                        ser.write(serial_cmd.encode('utf-8'))
                    
                    primary_object_processed = True # Ensure we only send one command per interval

                # Reset the baseline for the next 100ms interval
                obj_data["last_time"] = current_time
                obj_data["last_width"] = obj_data["smoothed_width"]

            # --- VISUALS ---
            display_color = (0, 255, 0) 
            if obj_data["status"] == "Moving Closer": display_color = (0, 0, 255)
            elif obj_data["status"] == "Moving Away": display_color = (255, 0, 0)

            # Draw deadzone guides (optional, helps with debugging calibration)
            cv2.line(frame, (int(frame_center_x - deadzone_pixels), 0), (int(frame_center_x - deadzone_pixels), frame_height), (255, 255, 255), 1)
            cv2.line(frame, (int(frame_center_x + deadzone_pixels), 0), (int(frame_center_x + deadzone_pixels), frame_height), (255, 255, 255), 1)

            # Draw box and status text
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), display_color, 2)
            cv2.putText(frame, f"ID {obj_id}: {obj_data['status']}", (int(x1), int(y1)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, display_color, 2)

    cv2.imshow("Direction Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if ser:
    ser.close()
cv2.destroyAllWindows()


# 3. Object Lock + Tracking
# Once detected:
# Assign an ID
# Switch to tracking mode
# Use:
# Deep SORT or a lighter tracker like:
# SORT

# 4. Throw Detection (this is critical)
# You need a state transition:
# HOLDING → THROWN → LANDED
# if distance(hand, object) increases rapidly AND object velocity > threshold:
#     state = THROWN

# 5. Trajectory Prediction (for camera movement)
# Once in THROWN state:
# Track object center across frames
# Fit a parabola (projectile motion)
# Use:
# Basic physics model (constant gravity)
# Or just fit a quadratic curve to points:
# x(t), y(t)
# You don’t need perfect physics—just a short-term prediction.

# 6. Moving Camera Control
# camera is stuck to the robot, so view changes as robot moves towards the predicted location
# Target = predicted landing point
# Approaches:
# Option B: Mobile Camera (robot)
# Convert image coords → world coords
# Use:
# Homography (if plane known)
# Or depth (stereo / RGB-D)
# Then:
# predicted landing → move robot → re-acquire object


# ⚠️ Hard Parts (don’t underestimate these)
# 1. Re-identification after occlusion
# If the object disappears mid-flight:
# Kalman filter prediction (built into SORT/DeepSORT)
# Keep track alive for ~0.5–1s without detection

# 2. Camera motion + tracking instability
# Moving camera = harder tracking
# Fix:
# Use image stabilization OR
# Track in camera frame, not world frame