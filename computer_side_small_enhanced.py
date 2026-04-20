import cv2
import zmq
import numpy as np
import traceback
import time  
from ultralytics import YOLO

# --- MOVEMENT & PROCESSING CONFIGURATION ---
COMMAND_COOLDOWN = 5.0  
EDGE_MARGIN = 50  # <-- NEW: Only process objects within this many pixels of the frame edges

print("Loading YOLO model...")
model = YOLO("best_medium_classes_removed.pt")

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")
print("Server listening on port 5555... Press 'q' in the video window to quit.")

last_command_time = 0.0  

while True:
    try:
        # 1. Receive and decode
        image_bytes = socket.recv()
        jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(jpg_as_np, flags=1)

        # 2. Guard against corrupted frames
        if frame is None:
            print("Warning: Received corrupted frame (imdecode returned None). Skipping.")
            socket.send_json({"detections": [], "command": None}) 
            continue

        # Get frame dimensions for grid logic and edge margins
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate 3x3 grid boundaries
        w_third = frame_width / 3
        h_third = frame_height / 3
        
        col_left_bound = w_third
        col_right_bound = 2 * w_third
        row_top_bound = h_third
        row_bottom_bound = 2 * h_third
        
        slow_margin_w = w_third / 2
        slow_margin_h = h_third / 2

        # 3. Run inference
        results = model(frame, verbose=False, device=0)
        detections = []
        
        serial_cmd = None
        primary_object_processed = False
        
        # 4. Extract data, evaluate position, and draw on the frame
        for r in results:
            if r.boxes is None:
                continue 
                
            for box in r.boxes:
                # Extract coordinates and info safely
                xyxy = box.xyxy[0].tolist() if box.xyxy is not None else [0, 0, 0, 0]
                conf = box.conf[0].item() if box.conf is not None else 0.0
                cls = int(box.cls[0].item()) if box.cls is not None else 0
                names_dict = r.names 
                
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                
                # --- NEW: Edge Filter Logic ---
                # Check if the bounding box is entirely inside the "ignore" zone.
                # If it is NOT touching the 50px margin on any side, skip it.
                if (x1 > EDGE_MARGIN and 
                    y1 > EDGE_MARGIN and 
                    x2 < (frame_width - EDGE_MARGIN) and 
                    y2 < (frame_height - EDGE_MARGIN)):
                    continue  # Skip this object completely
                
                if names_dict and cls in names_dict:
                    label = names_dict[cls]
                elif names_dict and str(cls) in names_dict:
                    label = names_dict[str(cls)]
                else:
                    label = f"ID_{cls}"
                
                # Calculate object centers for both axes
                obj_center_x = (x1 + x2) / 2
                obj_center_y = (y1 + y2) / 2

                if not primary_object_processed:
                    current_time = time.time()
                    desired_cmd = None
                    
                    # --- GRID NAVIGATION LOGIC ---
                    if obj_center_x < col_left_bound:
                        if obj_center_x > (col_left_bound - slow_margin_w):
                            desired_cmd = "slowrotateleft\n"
                        else:
                            desired_cmd = "rotateleft\n"
                            
                    elif obj_center_x > col_right_bound:
                        if obj_center_x < (col_right_bound + slow_margin_w):
                            desired_cmd = "slowrotateright\n"
                        else:
                            desired_cmd = "rotateright\n"
                            
                    elif obj_center_y > row_top_bound:
                        if obj_center_y < (row_top_bound + slow_margin_h):
                            desired_cmd = "slowforward\n"
                        else:
                            desired_cmd = "forward\n"
                            
                    else:
                        desired_cmd = "stop\n"
                        
                    # --- APPLY COMMAND & COOLDOWN ---
                    if desired_cmd == "stop\n":
                        serial_cmd = desired_cmd
                    elif (current_time - last_command_time) >= COMMAND_COOLDOWN:
                        serial_cmd = desired_cmd
                        last_command_time = current_time
                            
                    primary_object_processed = True

                # Add to our JSON list for the Pi
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "conf": round(conf, 2),
                    "class": label
                })

                # Draw bounding box and label on the PC frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label} {conf:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- Draw Visual Guides ---
        
        # 1. NEW: Draw the Edge Margin Border (Red)
        cv2.rectangle(frame, (EDGE_MARGIN, EDGE_MARGIN), 
                     (frame_width - EDGE_MARGIN, frame_height - EDGE_MARGIN), 
                     (0, 0, 255), 2)
        cv2.putText(frame, "IGNORE ZONE", (EDGE_MARGIN + 10, EDGE_MARGIN + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 2. Standard 3x3 Grid (White)
        cv2.line(frame, (int(col_left_bound), 0), (int(col_left_bound), frame_height), (255, 255, 255), 2)
        cv2.line(frame, (int(col_right_bound), 0), (int(col_right_bound), frame_height), (255, 255, 255), 2)
        cv2.line(frame, (0, int(row_top_bound)), (frame_width, int(row_top_bound)), (255, 255, 255), 2)
        cv2.line(frame, (0, int(row_bottom_bound)), (frame_width, int(row_bottom_bound)), (255, 255, 255), 2)

        # 3. Slow Zone Boundaries (Cyan)
        cv2.line(frame, (int(col_left_bound - slow_margin_w), 0), (int(col_left_bound - slow_margin_w), frame_height), (255, 255, 0), 1)
        cv2.line(frame, (int(col_right_bound + slow_margin_w), 0), (int(col_right_bound + slow_margin_w), frame_height), (255, 255, 0), 1)
        cv2.line(frame, (0, int(row_top_bound + slow_margin_h)), (frame_width, int(row_top_bound + slow_margin_h)), (255, 255, 0), 1)

        # 4. Highlight Target Block in Green
        cv2.rectangle(frame, (int(col_left_bound), 0), (int(col_right_bound), int(row_top_bound)), (0, 255, 0), 3)
        cv2.putText(frame, "TARGET", (int(col_left_bound) + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # --- Draw active command OR Cooldown Timer ---
        time_since_last_cmd = time.time() - last_command_time
        
        if serial_cmd:
            cv2.putText(frame, f"CMD: {serial_cmd.strip()}", (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif time_since_last_cmd < COMMAND_COOLDOWN:
            time_remaining = COMMAND_COOLDOWN - time_since_last_cmd
            cv2.putText(frame, f"WAIT: {time_remaining:.1f}s", (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Show the frame on the PC
        cv2.imshow("PC YOLO Inference Server", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting server...")
            socket.send_json({"detections": [], "command": None}) 
            break

        # 5. Send data and the string command back to the Pi
        payload = {
            "detections": detections,
            "command": serial_cmd
        }
        socket.send_json(payload)

    except Exception as e:
        print("\n--- ERROR CAUGHT ON PC SERVER ---")
        traceback.print_exc()
        print("---------------------------------\n")
        socket.send_json({"detections": [], "command": None})

cv2.destroyAllWindows()