import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import deque

# ========================
#  PREPROCESSING
# ========================
# Keep history of frames for temporal smoothing
_frame_history = deque(maxlen=5)

def preprocess_frame(frame, is_thermal=False, clahe_clip=2.0, tile_grid=(8,8), gamma=1.2, use_retinex=True):
    """
    Advanced preprocessing for thermal or RGB images.
    Args:
        frame: input frame (thermal grayscale or webcam BGR)
        is_thermal: set True if using thermal camera
    """

    # Step 1: Normalize if not 8-bit
    if frame.dtype != "uint8":
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    # Step 2: Convert to grayscale
    if is_thermal:
        gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 3: CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=tile_grid)
    frame = clahe.apply(gray)

    # Step 4: Gamma correction
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    frame = cv2.LUT(frame, table)

    # Step 5: Retinex (optional, helps fog/haze but heavy)
    if use_retinex:
        frame = multi_scale_retinex(frame)

    # Step 6: Temporal smoothing
    _frame_history.append(frame)
    if len(_frame_history) > 1:
        frame = np.mean(_frame_history, axis=0).astype("uint8")

    # Step 7: Denoising
    frame = cv2.fastNlMeansDenoising(frame, None, 10, 7, 21)

    # Step 8: Convert back to BGR for YOLO
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    return frame



def multi_scale_retinex(img, sigma_list=[15, 80, 250]):
    """ Multi-Scale Retinex (MSR) for haze/fog robustness """
    img = img.astype(np.float32) + 1.0
    retinex = np.zeros_like(img)

    for sigma in sigma_list:
        blur = cv2.GaussianBlur(img, (0,0), sigma)
        retinex += np.log10(img) - np.log10(blur + 1)

    retinex = retinex / len(sigma_list)

    # Normalize
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
    return retinex.astype("uint8")


# ========================
#  DETECTION + TRACKING
# ========================
def init_models(model_path="thermal1.pt", device="cpu"):
    detector = YOLO(model_path)
    tracker = DeepSort(max_age=15, n_init=3, max_cosine_distance=0.4)
    return detector, tracker


def detect_and_track(frame, detector, tracker, conf_thres=0.3):
    """
    Run YOLO detection + DeepSORT tracking on one frame.
    Returns: frame_with_boxes, list_of_tracks
    """
    results = detector(frame, verbose=False)[0]

    detections = []
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < conf_thres:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        cls_name = detector.names[cls_id]
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_name))

    tracks = tracker.update_tracks(detections, frame=frame)
    return tracks


# ========================
#  DRAWING
# ========================
def draw_tracks(frame, tracks):
    for t in tracks:
        if not t.is_confirmed():
            continue
        l, t_, r, b = map(int, t.to_ltrb())
        obj_id = t.track_id
        label = t.get_det_class() if hasattr(t, "get_det_class") else "object"

        cv2.rectangle(frame, (l, t_), (r, b), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label}-{obj_id}",
            (l, t_ - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return frame


# ========================
#  LOGGING
# ========================
def init_logger(log_dir="logs", log_file="detections.log"):
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, log_file)


def log_detection(log_path, obj_id, obj_type):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] ID={obj_id} | type={obj_type}"
    print(line)
    with open(log_path, "a") as f:
        f.write(line + "\n")


# ========================
#  CROPPING
# ========================
def save_crop(frame, bbox, obj_id, obj_type, save_dir="logs/crops"):
    os.makedirs(save_dir, exist_ok=True)
    l, t, r, b = map(int, bbox)
    crop = frame[t:b, l:r]
    if crop.size > 0:
        filename = f"{obj_type}_{obj_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(os.path.join(save_dir, filename), crop)
