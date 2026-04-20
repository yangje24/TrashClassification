import cv2
import zmq
import numpy as np
import traceback
import time
import os
from datetime import datetime

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
from sort import Sort

import torch
import clip as clip_model
from PIL import Image as PILImage


# =========================================================
# CONFIG
# =========================================================
ZMQ_PORT = 5556
TRASH_MODEL_PATH = "models/best_ncnn_model/best.pt"        # Trash detection YOLO (from webcam_inferance)
HAND_LANDMARK_MODEL = "hand_landmarker.task"

USE_CLIP = True
SAVE_CLIP_CROPS = False

CLIP_LABELS = [
    # Plastic
    "plastic bottle", "plastic bag", "plastic cap",
    # Paper & Cardboard
    "paper", "cardboard", "paper bag",
    # Metal
    "can", "aluminum can", "tin can", "aluminum foil", "bottle cap",
    # General / Mixed
    "garbage bag", "trash"
]

# Hand-to-trash detection ROI
HAND_ROI_SCALE = 1.35
TRASH_YOLO_CONF = 0.70

# Tracking
TRACK_MAX_AGE = 10
TRACK_MIN_HITS = 3
TRACK_IOU_THRESHOLD = 0.30

# Movement / command logic
COMMAND_COOLDOWN = 1.0
EDGE_MARGIN = 50


# =========================================================
# MediaPipe Tasks API setup
# =========================================================
HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS
HANDEDNESS_TEXT_COLOR = (88, 205, 54)
MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1

base_options = python.BaseOptions(model_asset_path=HAND_LANDMARK_MODEL)
hand_options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(hand_options)


# =========================================================
# Models and trackers
# =========================================================

# 1. Trash YOLO (hand-region, from webcam_inferance)
print(f"Loading trash YOLO model from {TRASH_MODEL_PATH} ...")
trash_model = YOLO(TRASH_MODEL_PATH, task="detect")

# 2. SORT tracker (for trash detections)
tracker = Sort(
    max_age=TRACK_MAX_AGE,
    min_hits=TRACK_MIN_HITS,
    iou_threshold=TRACK_IOU_THRESHOLD,
)
track_memory = {}  # track_id -> dict with stored class info

# 3. CLIP classifier
CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if USE_CLIP:
    print(f"Loading CLIP (ViT-B/32) on {CLIP_DEVICE} ...")
    clip_net, clip_preprocess = clip_model.load("ViT-B/32", device=CLIP_DEVICE)
    clip_tokens = clip_model.tokenize(CLIP_LABELS).to(CLIP_DEVICE)
    print("CLIP loaded")
else:
    clip_net = clip_preprocess = clip_tokens = None


# =========================================================
# Helpers
# =========================================================
def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw hand landmarks and handedness using MediaPipe Tasks outputs."""
    annotated_image = np.copy(rgb_image)
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        h, w, _ = annotated_image.shape

        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated_image, (cx, cy), 4, (0, 255, 0), -1)

        for connection in HAND_CONNECTIONS:
            start_lm = hand_landmarks[connection.start]
            end_lm = hand_landmarks[connection.end]
            start_pt = (int(start_lm.x * w), int(start_lm.y * h))
            end_pt = (int(end_lm.x * w), int(end_lm.y * h))
            cv2.line(annotated_image, start_pt, end_pt, (255, 0, 0), 2)

        x_coordinates = [lm.x for lm in hand_landmarks]
        y_coordinates = [lm.y for lm in hand_landmarks]
        text_x = int(min(x_coordinates) * w)
        text_y = int(min(y_coordinates) * h) - MARGIN

        cv2.putText(
            annotated_image,
            f"{handedness[0].category_name}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated_image


def hand_landmarks_to_box(image_shape, hand_landmarks, scale=1.0):
    h, w = image_shape[:2]
    xs = [int(lm.x * w) for lm in hand_landmarks]
    ys = [int(lm.y * h) for lm in hand_landmarks]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    cx = (xmin + xmax) // 2
    cy = (ymin + ymax) // 2
    box_w = max(1, xmax - xmin)
    box_h = max(1, ymax - ymin)

    new_w = int(box_w * scale)
    new_h = int(box_h * scale)

    x1 = max(0, cx - new_w // 2)
    y1 = max(0, cy - new_h // 2)
    x2 = min(w, cx + new_w // 2)
    y2 = min(h, cy + new_h // 2)
    return x1, y1, x2, y2


def crop_from_box(image, box):
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2].copy()


def union_boxes(a, b):
    if a is None:
        return b
    if b is None:
        return a
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)


def pad_box(box, image_shape, pad_ratio=0.2):
    x1, y1, x2, y2 = box
    h, w = image_shape[:2]
    pad_w = int((x2 - x1) * pad_ratio)
    pad_h = int((y2 - y1) * pad_ratio)
    return (
        max(0, x1 - pad_w),
        max(0, y1 - pad_h),
        min(w, x2 + pad_w),
        min(h, y2 + pad_h),
    )


def clip_classify(crop_bgr):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_image = PILImage.fromarray(crop_rgb)
    image_input = clip_preprocess(pil_image).unsqueeze(0).to(CLIP_DEVICE)

    with torch.no_grad():
        logits_per_image, _ = clip_net(image_input, clip_tokens)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    ranked = sorted(zip(CLIP_LABELS, probs), key=lambda x: x[1], reverse=True)
    best_label, best_prob = ranked[0]
    return best_label, best_prob, ranked


def choose_nearest_hand_box(obj_box, hand_boxes):
    if not hand_boxes:
        return None

    x1, y1, x2, y2 = obj_box
    obj_cx = (x1 + x2) / 2.0
    obj_cy = (y1 + y2) / 2.0

    best_box = None
    best_dist = float("inf")
    for hb in hand_boxes:
        hx1, hy1, hx2, hy2 = hb
        hcx = (hx1 + hx2) / 2.0
        hcy = (hy1 + hy2) / 2.0
        dist = (hcx - obj_cx) ** 2 + (hcy - obj_cy) ** 2
        if dist < best_dist:
            best_dist = dist
            best_box = hb
    return best_box


# =========================================================
# ZMQ server
# =========================================================
print("Starting PC server...")
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(f"tcp://*:{ZMQ_PORT}")
print(f"Server listening on port {ZMQ_PORT}... Press 'q' in the video window to quit.")

last_command_time = 0.0
timestamp_ms = 0
hand_tracking_active = False  # Only start following hand after first trash detection


# =========================================================
# Main loop
# =========================================================
while True:
    try:
        image_bytes = socket.recv()
        jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(jpg_as_np, flags=1)

        if frame is None:
            print("Warning: Received corrupted frame (imdecode returned None). Skipping.")
            socket.send_json({"detections": [], "command": None, "trash": []})
            continue

        frame_height, frame_width = frame.shape[:2]

        # 3x3 grid boundaries (for navigation)
        w_third = frame_width / 3
        h_third = frame_height / 3
        col_left_bound = w_third
        col_right_bound = 2 * w_third
        row_top_bound = h_third
        row_bottom_bound = 2 * h_third
        slow_margin_w = w_third / 2
        slow_margin_h = h_third / 2

        serial_cmd = None
        primary_cmd_processed = False

        # =============================================================
        # Hand detection + Trash YOLO + CLIP + Navigation
        # =============================================================
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        timestamp_ms += 33
        detection_result = detector.detect_for_video(mp_image, timestamp_ms)

        # Draw hand landmarks onto the RGB image
        annotated_rgb = draw_landmarks_on_image(image_rgb, detection_result)

        hand_boxes = []
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                hand_boxes.append(
                    hand_landmarks_to_box(frame.shape, hand_landmarks, scale=1.0))

        # Navigation: follow the first detected hand (only after first trash detection)
        if hand_tracking_active and hand_boxes and not primary_cmd_processed:
            hx1, hy1, hx2, hy2 = hand_boxes[0]
            hand_cx = (hx1 + hx2) / 2
            hand_cy = (hy1 + hy2) / 2

            if hand_cx < col_left_bound:
                if hand_cx > (col_left_bound - slow_margin_w):
                    desired_cmd = "rotateleftslow\n"
                else:
                    desired_cmd = "rotateleft\n"
            elif hand_cx > col_right_bound:
                if hand_cx < (col_right_bound + slow_margin_w):
                    desired_cmd = "rotaterightslow\n"
                else:
                    desired_cmd = "rotateright\n"
            elif hand_cy > row_top_bound:
                if hand_cy < (row_top_bound + slow_margin_h):
                    desired_cmd = "forwardslow\n"
                else:
                    desired_cmd = "forward\n"
            else:
                desired_cmd = "stop\n"

            current_time = time.time()
            if desired_cmd == "stop\n":
                serial_cmd = desired_cmd
            elif (current_time - last_command_time) >= COMMAND_COOLDOWN:
                serial_cmd = desired_cmd
                last_command_time = current_time

            primary_cmd_processed = True

        # Run trash YOLO only near detected hands
        all_trash_detections = []
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                roi_box = hand_landmarks_to_box(
                    frame.shape, hand_landmarks, scale=HAND_ROI_SCALE)
                roi = crop_from_box(frame, roi_box)
                if roi is None or roi.size == 0:
                    continue

                results = trash_model.predict(
                    source=roi, show=False, conf=TRASH_YOLO_CONF,
                    verbose=False)
                offset_x, offset_y = roi_box[0], roi_box[1]

                for box in results[0].boxes:
                    tx1, ty1, tx2, ty2 = box.xyxy[0].cpu().numpy()
                    score = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())

                    tx1 += offset_x
                    tx2 += offset_x
                    ty1 += offset_y
                    ty2 += offset_y

                    all_trash_detections.append(
                        [tx1, ty1, tx2, ty2, score, cls_id])

        # Activate hand tracking on first trash detection
        if all_trash_detections and not hand_tracking_active:
            hand_tracking_active = True
            print("Trash detected in hand — hand tracking activated!")

        # SORT tracking on trash detections
        if len(all_trash_detections) > 0:
            dets_np = np.array(
                [d[:5] for d in all_trash_detections], dtype=np.float32)
        else:
            dets_np = np.empty((0, 5), dtype=np.float32)

        tracked_objects = tracker.update(dets_np)

        trash_info_list = []

        for obj in tracked_objects:
            tx1, ty1, tx2, ty2, track_id = [int(v) for v in obj]
            tx1 = max(0, tx1)
            ty1 = max(0, ty1)
            tx2 = min(frame_width, tx2)
            ty2 = min(frame_height, ty2)

            if tx2 <= tx1 or ty2 <= ty1:
                continue

            # One-time CLIP classification per track
            if track_id not in track_memory:
                best_hand_box = choose_nearest_hand_box(
                    (tx1, ty1, tx2, ty2), hand_boxes)
                crop_box = union_boxes(
                    (tx1, ty1, tx2, ty2), best_hand_box)
                crop_box = pad_box(crop_box, frame.shape, pad_ratio=0.2)
                crop = crop_from_box(frame, crop_box)

                if crop is not None and crop.size > 0 and USE_CLIP:
                    if SAVE_CLIP_CROPS:
                        os.makedirs("clip_crops", exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        crop_filename = (
                            f"clip_crops/track_{track_id}_{timestamp}.jpg")
                        cv2.imwrite(crop_filename, crop)

                    best_label, best_prob, ranked = clip_classify(crop)
                    track_memory[track_id] = {
                        "yolo": "trash",
                        "clip_label": best_label,
                        "clip_prob": float(best_prob),
                        "clip_top3": ranked[:3],
                    }
                    print(f"\n-- Track {track_id} NEW --")
                    print(f"  CLIP top-1: {best_label:<30s} {best_prob:6.2%}")
                else:
                    track_memory[track_id] = {
                        "yolo": "trash",
                        "clip_label": "unknown",
                        "clip_prob": 0.0,
                        "clip_top3": [],
                    }

            info = track_memory[track_id]
            label_text = (
                f"ID{track_id}: {info['clip_label']}"
                f" ({info['clip_prob']:.0%})")

            trash_info_list.append({
                "track_id": int(track_id),
                "box": [tx1, ty1, tx2, ty2],
                "clip_label": info["clip_label"],
                "clip_prob": round(info["clip_prob"], 2),
            })


            # Draw trash bounding box (yellow) and label on RGB image
            cv2.rectangle(annotated_rgb, (tx1, ty1), (tx2, ty2),
                          (0, 255, 255), 2)
            (tw, th), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_rgb, (tx1, ty1 - th - 8),
                          (tx1 + tw, ty1), (0, 255, 255), -1)
            cv2.putText(annotated_rgb, label_text, (tx1, ty1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # =============================================================
        # PART C: Draw visual guides (from computer_side_small)
        # =============================================================

        # Convert annotated RGB back to BGR for display, merging hand+trash drawings
        display_frame = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        # Edge margin border (red)
        cv2.rectangle(
            display_frame,
            (EDGE_MARGIN, EDGE_MARGIN),
            (frame_width - EDGE_MARGIN, frame_height - EDGE_MARGIN),
            (0, 0, 255), 2)
        cv2.putText(
            display_frame, "IGNORE ZONE",
            (EDGE_MARGIN + 10, EDGE_MARGIN + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 3x3 grid (white)
        cv2.line(display_frame, (int(col_left_bound), 0),
                 (int(col_left_bound), frame_height), (255, 255, 255), 2)
        cv2.line(display_frame, (int(col_right_bound), 0),
                 (int(col_right_bound), frame_height), (255, 255, 255), 2)
        cv2.line(display_frame, (0, int(row_top_bound)),
                 (frame_width, int(row_top_bound)), (255, 255, 255), 2)
        cv2.line(display_frame, (0, int(row_bottom_bound)),
                 (frame_width, int(row_bottom_bound)), (255, 255, 255), 2)

        # Slow zone boundaries (cyan)
        cv2.line(display_frame,
                 (int(col_left_bound - slow_margin_w), 0),
                 (int(col_left_bound - slow_margin_w), frame_height),
                 (255, 255, 0), 1)
        cv2.line(display_frame,
                 (int(col_right_bound + slow_margin_w), 0),
                 (int(col_right_bound + slow_margin_w), frame_height),
                 (255, 255, 0), 1)
        cv2.line(display_frame,
                 (0, int(row_top_bound + slow_margin_h)),
                 (frame_width, int(row_top_bound + slow_margin_h)),
                 (255, 255, 0), 1)

        # Target block highlight (green)
        cv2.rectangle(
            display_frame,
            (int(col_left_bound), 0),
            (int(col_right_bound), int(row_top_bound)),
            (0, 255, 0), 3)
        cv2.putText(
            display_frame, "TARGET",
            (int(col_left_bound) + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw active command
        if serial_cmd:
            cv2.putText(
                display_frame, f"CMD: {serial_cmd.strip()}",
                (10, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Show frame
        cv2.imshow("PC Combined Server", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting server...")
            socket.send_json({
                "detections": [], "command": None, "trash": []})
            break

        # =============================================================
        # Send combined payload back to Pi
        # =============================================================
        payload = {
            "detections": trash_info_list,  # Tracked trash with CLIP labels
            "command": serial_cmd,          # Movement command based on trash position
            "trash": trash_info_list,       # Same list (kept for Pi compatibility)
        }
        print(f"Sending payload: {payload}")
        socket.send_json(payload)

    except Exception:
        print("\n--- ERROR CAUGHT ON PC SERVER ---")
        traceback.print_exc()
        print("---------------------------------\n")
        try:
            socket.send_json({
                "detections": [], "command": None, "trash": []})
        except Exception:
            pass

cv2.destroyAllWindows()
