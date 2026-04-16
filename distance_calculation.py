import cv2
import time
from ultralytics import YOLO

# --- CONFIGURATION ---
CONF_LEVEL = 0.3 
UPDATE_INTERVAL = 0.1  # 100 ms interval for checking movement
JITTER_THRESHOLD = 2.0 # Minimum pixel change required to trigger a status update
SMOOTHING_FACTOR = 0.4 # Between 0 and 1. Lower = smoother but slightly delayed

model = YOLO("best_ncnn_model") 
cap = cv2.VideoCapture(0)

# object_memory[id] stores: {"last_time": float, "last_width": float, "smoothed_width": float, "status": str}
object_memory = {}

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    current_time = time.time()
    results = model.track(frame, persist=True, conf=CONF_LEVEL, verbose=False)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().numpy()
        
        for i, obj_id in enumerate(ids):
            x1, y1, x2, y2 = boxes[i]
            pixel_width = x2 - x1
            
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
                # Compare the smoothed width from now vs 100ms ago
                width_diff = obj_data["smoothed_width"] - obj_data["last_width"]
                
                if width_diff > JITTER_THRESHOLD:
                    obj_data["status"] = "Moving Closer"
                    color = (0, 0, 255) # Red
                elif width_diff < -JITTER_THRESHOLD:
                    obj_data["status"] = "Moving Away"
                    color = (255, 0, 0) # Blue
                else:
                    obj_data["status"] = "Stationary / Lateral"
                    color = (0, 255, 0) # Green
                
                # Reset the baseline for the next 100ms interval
                obj_data["last_time"] = current_time
                obj_data["last_width"] = obj_data["smoothed_width"]

            # --- VISUALS ---
            # Default color fallback if status hasn't evaluated yet
            display_color = (0, 255, 0) 
            if obj_data["status"] == "Moving Closer": display_color = (0, 0, 255)
            elif obj_data["status"] == "Moving Away": display_color = (255, 0, 0)

            # Draw box and status text
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), display_color, 2)
            cv2.putText(frame, f"ID {obj_id}: {obj_data['status']}", (int(x1), int(y1)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, display_color, 2)

    cv2.imshow("Direction Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()