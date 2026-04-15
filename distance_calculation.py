import cv2
import time
from collections import deque
from ultralytics import YOLO

# --- CONFIGURATION ---
FOCAL_LENGTH = 600 
REAL_WIDTH = 6 
FRAME_GAP = 10  
CONF_LEVEL = 0.3 

model = YOLO("best_ncnn_model") 
cap = cv2.VideoCapture(0)

# object_memory[id] stores: {"history": deque, "max_speed": float}
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
            
            current_dist = (REAL_WIDTH * FOCAL_LENGTH) / pixel_width
            obj_id = int(obj_id)

            if obj_id not in object_memory:
                object_memory[obj_id] = {
                    "history": deque(maxlen=FRAME_GAP + 1),
                    "max_speed": 0.0
                }
            
            obj_data = object_memory[obj_id]
            obj_data["history"].append((current_dist, current_time))

            current_speed = 0
            if len(obj_data["history"]) > FRAME_GAP:
                old_dist, old_time = obj_data["history"][0]
                delta_d = abs(old_dist - current_dist) 
                delta_t = current_time - old_time
                
                if delta_t > 0:
                    current_speed = delta_d / delta_t
                    if current_speed > obj_data["max_speed"]:
                        obj_data["max_speed"] = current_speed

            # --- PROJECTION ---
            projected_travel = obj_data["max_speed"] * 1.0 

            # --- PRINT STATEMENT ---
            if projected_travel > 0:
                print(f"[ID {obj_id}] Current Dist: {current_dist:.1f}cm | Peak Spd: {obj_data['max_speed']:.1f}cm/s | Est. Travel (1s): {projected_travel:.1f}cm")

            # --- VISUALS ---
            color = (255, 0, 255) 
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"Proj Travel: {projected_travel:.1f}cm", (int(x1), int(y1)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imshow("Peak Speed Projection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()