import mediapipe as mp
import numpy as np
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

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

    # Convert normalized → pixel coordinates
    xs = [int(lm.x * w) for lm in hand_landmarks]
    ys = [int(lm.y * h) for lm in hand_landmarks]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # Compute center + size
    cx = (xmin + xmax) // 2
    cy = (ymin + ymax) // 2
    box_w = xmax - xmin
    box_h = ymax - ymin

    # Expand box
    new_w = int(box_w * scale)
    new_h = int(box_h * scale)

    new_xmin = max(0, cx - new_w // 2)
    new_xmax = min(w, cx + new_w // 2)
    new_ymin = max(0, cy - new_h // 2)
    new_ymax = min(h, cy + new_h // 2)

    return image[new_ymin:new_ymax, new_xmin:new_xmax]


base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# 2. ROI

MODEL_PATH = "trash_binary/best.pt"#"best_ncnn_model"#"testmod/best.pt"

print("Loading model from", MODEL_PATH)
model = YOLO(MODEL_PATH, task="detect")

# end 2

print("Starting webcam inference. Press 'q' to quit.")

cap = cv2.VideoCapture(0)
try:
    # For webcam input:
    while True: #cap.isOpened():
        success, image = cap.read()
        if not success or image is None:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        # image.flags.writeable = True
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # mp_image = mp.Image.create_from_file(image)
        detection_result = detector.detect(mp_image)
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
        cv2.imshow('MediaPipe Hands', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))#flip(image, 1))

        # if hand present
        if len(detection_result.hand_landmarks)!=0:
            # crop region near hand
            # roi = image_rgb
            # # run trash model
            # results = model.predict(source=roi, show=False, conf=0.7)
            # annotated_frame = results[0].plot()


            for hand_landmarks in detection_result.hand_landmarks:
                roi = get_expanded_hand_roi(annotated_image, hand_landmarks, scale=3)

                if roi.size == 0:
                    continue

                results = model.predict(source=roi, show=False, conf=0.7)
                annotated_frame = results[0].plot()

                cv2.imshow("Webcam Trash Detection", annotated_frame)


            # cv2.imshow("Webcam Trash Detection", annotated_frame)
    
        if cv2.waitKey(5) & 0xFF == 27:
            break

        

finally:    
    cap.release()
    cv2.destroyAllWindows()