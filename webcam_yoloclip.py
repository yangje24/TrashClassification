import os
from datetime import datetime
import mediapipe as mp
import numpy as np
import cv2
import torch
import clip as clip_model
from PIL import Image as PILImage
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

# ── CLIP labels for detailed trash sub-classification ──────────────────────
CLIP_LABELS = [
    # Plastic
    "plastic bottle", "plastic bag", "plastic cap",
    
    # Paper & Cardboard
    "paper", "cardboard", "paper bag", 
    
    # Metal
    "can", "aluminum can", "tin can", "aluminum foil", "bottle cap"
    
    # General / Mixed
    "garbage bag", "trash"
]


# ── Helper functions ───────────────────────────────────────────────────────

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


def clip_classify(crop_bgr, clip_net, clip_preprocess, clip_tokens, device):
    """Run CLIP zero-shot classification on a BGR numpy crop.
    
    Returns (best_label, best_prob, all_probs) where all_probs is a list of
    (label, probability) tuples sorted descending.
    """
    # Convert BGR numpy → RGB PIL
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_image = PILImage.fromarray(crop_rgb)

    image_input = clip_preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_per_image, _ = clip_net(image_input, clip_tokens)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    ranked = sorted(zip(CLIP_LABELS, probs), key=lambda x: x[1], reverse=True)
    best_label, best_prob = ranked[0]
    return best_label, best_prob, ranked


# ── Model setup ────────────────────────────────────────────────────────────

# 1. MediaPipe hand detector
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# 2. YOLO trash detector
MODEL_PATH = "models/best_ncnn_model/best.pt"
print("Loading YOLO model from", MODEL_PATH)
yolo_model = YOLO(MODEL_PATH, task="detect")

# 3. CLIP classifier
CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CLIP (ViT-B/32) on {CLIP_DEVICE}...")
clip_net, clip_preprocess = clip_model.load("ViT-B/32", device=CLIP_DEVICE)
clip_tokens = clip_model.tokenize(CLIP_LABELS).to(CLIP_DEVICE)
print("CLIP loaded ✅")

# 4. SORT tracker
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)
id_to_class = {}       # track_id → (yolo_class, clip_label, clip_prob)

# ── Main loop ──────────────────────────────────────────────────────────────

print("Starting webcam inference. Press ESC to quit.")

cap = cv2.VideoCapture(0)
try:
    while True:
        success, image = cap.read()
        if not success or image is None:
            print("Ignoring empty camera frame.")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Hand detection
        detection_result = detector.detect(mp_image)
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)

        # If hand(s) present → look for trash near each hand
        if len(detection_result.hand_landmarks) != 0:
            all_detections = []   # accumulate detections across all hands

            for hand_landmarks in detection_result.hand_landmarks:
                # Use a smaller scale (1.3) so YOLO ONLY sees objects right inside/near the hand.
                # Background objects outside this small bounding box are completely ignored.
                roi, offset_x, offset_y = get_expanded_hand_roi(image, hand_landmarks, scale=1.3)

                if roi.size == 0:
                    continue

                results = yolo_model.predict(source=roi, show=False, conf=0.7, verbose=False)

                # Collect detections (convert ROI coords → full-frame coords)
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    score = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())

                    # Offset to global image coordinates
                    x1 += offset_x
                    x2 += offset_x
                    y1 += offset_y
                    y2 += offset_y

                    all_detections.append([x1, y1, x2, y2, score, cls_id])

            # Run SORT tracker on accumulated detections
            if len(all_detections) > 0:
                dets_np = np.array([d[:5] for d in all_detections])
                # Build a map from detection index → yolo class
                det_classes = {i: d[5] for i, d in enumerate(all_detections)}
            else:
                dets_np = np.empty((0, 5))
                det_classes = {}

            tracked_objects = tracker.update(dets_np)

            for obj in tracked_objects:
                x1, y1, x2, y2, track_id = obj.astype(int)

                # Clamp to image bounds
                h_img, w_img = image.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w_img, x2)
                y2 = min(h_img, y2)

                # If this is a NEW track → run CLIP on the crop
                if track_id not in id_to_class:
                    # We want the crop to include BOTH the object AND the hand.
                    # Find the hand closest to the object and merge their bounding boxes.
                    clip_x1, clip_y1, clip_x2, clip_y2 = x1, y1, x2, y2
                    
                    best_hand_box = None
                    min_dist = float('inf')
                    obj_cx = (x1 + x2) / 2
                    obj_cy = (y1 + y2) / 2
                    
                    for hand_landmarks in detection_result.hand_landmarks:
                        hxs = [int(lm.x * w_img) for lm in hand_landmarks]
                        hys = [int(lm.y * h_img) for lm in hand_landmarks]
                        hx1, hx2 = min(hxs), max(hxs)
                        hy1, hy2 = min(hys), max(hys)
                        hcx, hcy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
                        dist = (hcx - obj_cx)**2 + (hcy - obj_cy)**2
                        if dist < min_dist:
                            min_dist = dist
                            best_hand_box = (hx1, hy1, hx2, hy2)
                            
                    if best_hand_box:
                        hx1, hy1, hx2, hy2 = best_hand_box
                        # Union of the object box and the nearest hand box
                        clip_x1 = min(x1, hx1)
                        clip_y1 = min(y1, hy1)
                        clip_x2 = max(x2, hx2)
                        clip_y2 = max(y2, hy2)
                    
                    # Add 20% padding around the union box to ensure no edges are cut
                    pad_w = int((clip_x2 - clip_x1) * 0.2)
                    pad_h = int((clip_y2 - clip_y1) * 0.2)
                    
                    clip_x1 = max(0, clip_x1 - pad_w)
                    clip_y1 = max(0, clip_y1 - pad_h)
                    clip_x2 = min(w_img, clip_x2 + pad_w)
                    clip_y2 = min(h_img, clip_y2 + pad_h)

                    crop = image[clip_y1:clip_y2, clip_x1:clip_x2]
                    
                    if crop.size > 0:
                        # Save the crop
                        os.makedirs("clip_crops", exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        crop_filename = f"clip_crops/track_{track_id}_{timestamp}.jpg"
                        cv2.imwrite(crop_filename, crop)

                        best_label, best_prob, ranked = clip_classify(
                            crop, clip_net, clip_preprocess, clip_tokens, CLIP_DEVICE
                        )
                        # Get YOLO class name from the model
                        yolo_cls_name = "trash"
                        id_to_class[track_id] = {
                            "yolo": yolo_cls_name,
                            "clip_label": best_label,
                            "clip_prob": best_prob,
                            "clip_top3": ranked[:3],
                        }
                        print(f"\n── Track {track_id} NEW ──")
                        print(f"  Saved crop: {crop_filename}")
                        print(f"  CLIP top-1:")
                        for lbl, p in ranked[:1]:
                            print(f"    {lbl:<30s} {p:6.2%}")
                    else:
                        id_to_class[track_id] = {
                            "yolo": "trash",
                            "clip_label": "unknown",
                            "clip_prob": 0.0,
                            "clip_top3": [],
                        }

                info = id_to_class[track_id]
                label_text = f"ID{track_id}: {info['clip_label']} ({info['clip_prob']:.0%})"

                # Draw bounding box + label on annotated image
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw background rectangle for text
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_image, (x1, y1 - th - 8), (x1 + tw, y1), (0, 255, 0), -1)
                cv2.putText(annotated_image, label_text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Show main window
        cv2.imshow('Trash Detection + CLIP', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(5) & 0xFF == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
