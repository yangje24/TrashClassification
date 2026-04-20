import cv2
import zmq
import numpy as np
import traceback
from ultralytics import YOLO

print("Loading YOLO model...")
model = YOLO("best.pt")


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")
print("Server listening on port 5555... Press 'q' in the video window to quit.")

while True:
    try:
        # 1. Receive and decode
        image_bytes = socket.recv()
        jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(jpg_as_np, flags=1)

        # 2. Guard against corrupted frames
        if frame is None:
            print("Warning: Received corrupted frame (imdecode returned None). Skipping.")
            socket.send_json([]) 
            continue

        # 3. Run inference
        results = model(frame, verbose=False, device=0)
        detections = []
        
        # 4. Extract data and draw on the frame
        for r in results:
            if r.boxes is None:
                continue 
                
            for box in r.boxes:
                # Extract coordinates and info safely
                xyxy = box.xyxy[0].tolist() if box.xyxy is not None else [0, 0, 0, 0]
                conf = box.conf[0].item() if box.conf is not None else 0.0
                cls = int(box.cls[0].item()) if box.cls is not None else 0
                # Get the names dictionary directly from the result object
                names_dict = r.names 
                
                # Check for integer key, then string key, then fallback to just printing the ID number
                if names_dict and cls in names_dict:
                    label = names_dict[cls]
                elif names_dict and str(cls) in names_dict:
                    label = names_dict[str(cls)]
                else:
                    label = f"ID_{cls}" # At least show the number so we know it's working!
                
                # Convert float coordinates to integers for OpenCV drawing
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                
                # Add to our JSON list for the Pi
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "conf": round(conf, 2),
                    "class": label
                })

                # --- NEW: Draw bounding box and label on the PC frame ---
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label} {conf:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- NEW: Show the frame on the PC ---
        cv2.imshow("PC YOLO Inference Server", frame)
        
        # Wait 1ms to allow the window to update, and check if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting server...")
            # Send a final empty reply so the Pi doesn't freeze waiting for this frame
            socket.send_json([]) 
            break

        # 5. Send data back to the Pi
        socket.send_json(detections)

    except Exception as e:
        print("\n--- ERROR CAUGHT ON PC SERVER ---")
        traceback.print_exc()
        print("---------------------------------\n")
        socket.send_json([])

# Clean up the window when the loop breaks
cv2.destroyAllWindows()