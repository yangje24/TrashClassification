import mediapipe as mp
import numpy as np
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
from sort import Sort

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    mp_drawing.draw_landmarks(
      annotated_image,
      hand_landmarks,
      mp_hands.HAND_CONNECTIONS,
      mp_drawing_styles.get_default_hand_landmarks_style(),
      mp_drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image
  
def get_expanded_hand_roi(image, hand_landmarks, scale=1.5):
    h, w, _ = image.shape

    xs = [int(lm.x * w) for lm in hand_landmarks]
    ys = [int(lm.y * h) for lm in hand_landmarks]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    cx = (xmin + xmax) // 2
    cy = (ymin + ymax) // 2

    box_w = xmax - xmin
    box_h = ymax - ymin

    new_w = int(box_w * scale)
    new_h = int(box_h * scale)

    new_xmin = max(0, cx - new_w // 2)
    new_xmax = min(w, cx + new_w // 2)
    new_ymin = max(0, cy - new_h // 2)
    new_ymax = min(h, cy + new_h // 2)

    roi = image[new_ymin:new_ymax, new_xmin:new_xmax]

    return roi, new_xmin, new_ymin

def get_hand_bbox(image, hand_landmarks):
    h, w, _ = image.shape
    xs = [int(lm.x * w) for lm in hand_landmarks]
    ys = [int(lm.y * h) for lm in hand_landmarks]
    return min(xs), min(ys), max(xs), max(ys)

def is_inside(box, hand_box):
    x1, y1, x2, y2 = box
    hx1, hy1, hx2, hy2 = hand_box

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    return hx1 <= cx <= hx2 and hy1 <= cy <= hy2

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# 2. ROI

MODEL_PATH = "best_ncnn_model"#"testmod/best.pt"

print("Loading model from", MODEL_PATH)
model = YOLO(MODEL_PATH, task="detect")
model.verbose = False

# 3. Object tracking

tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)
id_to_class = {}
# end 3

print("Starting webcam inference. Press 'q' to quit.")

cap = cv2.VideoCapture(0)
try:
    # --- STATE ---
    mode = "hand"
    tracked_id = None
    lost_frames = 0

    def get_center(x1, y1, x2, y2):
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    while True:
        success, image = cap.read()
        if not success or image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        detection_result = detector.detect(mp_image)

        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
        display_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        detections = []
        hand_boxes = []

        if len(detection_result.hand_landmarks) != 0:
            for hand_landmarks in detection_result.hand_landmarks:
                hand_boxes.append(get_hand_bbox(image, hand_landmarks))

        # =========================
        # 🟢 HAND MODE (ROI detect)
        # =========================
        if mode == "hand":
            if len(detection_result.hand_landmarks) != 0:
                for hand_landmarks in detection_result.hand_landmarks:
                    roi, offset_x, offset_y = get_expanded_hand_roi(image, hand_landmarks, scale=3)

                    if roi.size == 0:
                        continue

                    results = model.predict(source=roi, show=False, conf=0.6)

                    if len(results[0].boxes) == 0:
                        continue

                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        score = box.conf[0].cpu().numpy()

                        # convert to global
                        x1 += offset_x
                        x2 += offset_x
                        y1 += offset_y
                        y2 += offset_y

                        detections.append([x1, y1, x2, y2, score])

        # =========================
        # 🔴 TRACKING MODE (full frame)
        # =========================
        elif mode == "tracking":
            results = model.predict(source=image, show=False, conf=0.4)

            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                score = box.conf[0].cpu().numpy()
                detections.append([x1, y1, x2, y2, score])

        # =========================
        # 🧠 SORT TRACKING (always)
        # =========================
        if len(detections) > 0:
            dets_np = np.array(detections)
        else:
            dets_np = np.empty((0, 5))

        tracked_objects = tracker.update(dets_np)

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj.astype(int)
            box = (x1, y1, x2, y2)

            inside_any_hand = any(is_inside(box, hb) for hb in hand_boxes)

            if mode == "hand" and not inside_any_hand:
                print("THROW DETECTED → switching to tracking mode")
                mode = "tracking"
                tracked_id = track_id
        # =========================
        # 🎯 DRAW + THROW DETECTION
        # =========================
        # current_center = None
        
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj.astype(int)

            # In tracking mode, only draw the selected object
            if mode == "tracking" and track_id != tracked_id:
                continue

            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(display_image, f"ID {track_id}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # =========================
        # 🔄 RESET CONDITION
        # =========================
        if mode == "tracking":
            active_ids = [int(obj[4]) for obj in tracked_objects]

            if tracked_id not in active_ids:
                lost_frames += 1
            else:
                lost_frames = 0

            if lost_frames > 10:
                print("Object lost → back to HAND mode")
                mode = "hand"
                tracked_id = None
                lost_frames = 0

        # =========================
        # 📺 DISPLAY
        # =========================
        cv2.putText(display_image, f"MODE: {mode}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Final Output", display_image)

        if cv2.waitKey(5) & 0xFF == 27:
            break


    # Code without SORT tracking for throw detection:
    # # For webcam input:
    # while True: #cap.isOpened():
    #     success, image = cap.read()
    #     if not success or image is None:
    #         print("Ignoring empty camera frame.")
    #         # If loading a video, use 'break' instead of 'continue'.
    #         continue

    #     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    #     # image.flags.writeable = True
    #     # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #     # mp_image = mp.Image.create_from_file(image)
    #     detection_result = detector.detect(mp_image)
    #     annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
    #     display_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    #     cv2.imshow('MediaPipe Hands', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))#flip(image, 1))

    #     # if hand present
    #     if len(detection_result.hand_landmarks)!=0:
    #         # crop region near hand
    #         # roi = image_rgb
    #         # # run trash model
    #         # results = model.predict(source=roi, show=False, conf=0.7)
    #         # annotated_frame = results[0].plot()

    #         for hand_landmarks in detection_result.hand_landmarks:
    #             roi, offset_x, offset_y = get_expanded_hand_roi(image, hand_landmarks, scale=3)

    #             if roi.size == 0:
    #                 continue
    #             results = model.predict(source=roi, show=False, conf=0.7)
    #             annotated_frame = results[0].plot()
    #             cv2.imshow("Webcam Trash Detection", annotated_frame)

    #             print("Detections:", results[0].boxes)
    #             print(results)
                

    #             # ALWAYS define outside hand loop
    #             detections = []

    #             # --- DETECTION (ROI-based) ---
    #             if len(detection_result.hand_landmarks) != 0:
    #                 for hand_landmarks in detection_result.hand_landmarks:
    #                     roi, offset_x, offset_y = get_expanded_hand_roi(image, hand_landmarks, scale=3)

    #                     if roi.size == 0:
    #                         continue

    #                     results = model.predict(source=roi, show=False, conf=0.7)

    #                     if len(results[0].boxes) == 0:
    #                         continue

    #                     for box in results[0].boxes:
    #                         x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    #                         score = box.conf[0].cpu().numpy()

    #                         # Convert to global coords
    #                         x1 += offset_x
    #                         x2 += offset_x
    #                         y1 += offset_y
    #                         y2 += offset_y

    #                         detections.append([x1, y1, x2, y2, score])

    #             # --- TRACKING (ALWAYS RUN) ---
    #             if len(detections) > 0:
    #                 dets_np = np.array(detections)
    #             else:
    #                 dets_np = np.empty((0, 5))

    #             tracked_objects = tracker.update(dets_np)

    #             # --- DRAW ---
    #             for obj in tracked_objects:
    #                 x1, y1, x2, y2, track_id = obj.astype(int)

    #                 cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
    #                 cv2.putText(display_image, f"ID {track_id}", (x1, y1 - 10),
    #                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    #             # --- SHOW ONCE ---
    #             cv2.imshow('Final Output', display_image)
    #             # detections = []
    #             # for box in results[0].boxes:
    #             #     x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    #             #     score = box.conf[0].cpu().numpy()

    #             #     # Convert to global coordinates
    #             #     x1 += offset_x
    #             #     x2 += offset_x
    #             #     y1 += offset_y
    #             #     y2 += offset_y

    #             #     id = int(box.cls[0].cpu().numpy())
    #             #     print("ID:", id, "Score:", score)

    #             #     detections.append([x1, y1, x2, y2, score])
                
    #             # if len(detections) > 0:
    #             #     dets_np = np.array(detections)
    #             # else:
    #             #     dets_np = np.empty((0, 6))

    #             # tracked_objects = tracker.update(dets_np)

    #             # for obj in tracked_objects:
    #             #     print("Tracked object:", obj)
    #             #     print(obj.astype(int))
    #             #     x1, y1, x2, y2, track_id = obj.astype(int)

    #             #     if track_id not in id_to_class:
    #             #         id_to_class[track_id] = "trash" #results.names[cls_id]  # or from YOLO class

    #             #     label = id_to_class[track_id]

    #             #     cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #             #     cv2.putText(display_image, f"ID {track_id}", (x1, y1 - 10),
    #             #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
    #             # cv2.imshow('Final Output', display_image)


    #         # for hand_landmarks in detection_result.hand_landmarks:
    #         #     roi = get_expanded_hand_roi(image, hand_landmarks, scale=3)
    #         #     if roi.size == 0:
    #         #         continue
    #         #     results = model.predict(source=roi, show=False, conf=0.7)
    #         #     annotated_frame = results[0].plot()
    #         #     cv2.imshow("Webcam Trash Detection", annotated_frame)

    #             # if trash is classified, index


    #         # cv2.imshow("Webcam Trash Detection", annotated_frame)
    
    #     if cv2.waitKey(5) & 0xFF == 27:
    #         break

        

finally:    
    cap.release()
    cv2.destroyAllWindows()

# things to test:
# try w clip model
# track and move to hand position