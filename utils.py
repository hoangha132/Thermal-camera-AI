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

def preprocess_frame_fast(frame, is_thermal=False):
    """
    Fast preprocessing optimized for performance
    """
    if frame.dtype != "uint8":
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    if is_thermal:
        # Fast LAB equalization for thermal
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        lab = cv2.merge((l, a, b))
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return frame

def preprocess_raw_thermal_fast(frame):
    """
    Fast thermal preprocessing
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = 255 - gray  # Invert
    gray = cv2.equalizeHist(gray)
    out = cv2.merge([gray, gray, gray])
    return out

# ========================
#  DETECTION + TRACKING (OPTIMIZED)
# ========================
def init_models(model_path="thermal1.pt", device="cpu"):
    """
    Initialize models with optimizations
    """
    # Use GPU if available
    if device == "auto":
        device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    
    print(f"🚀 Using device: {device}")
    detector = YOLO(model_path)
    
    # Correct DeepSort parameters
    tracker = DeepSort(
        max_age=30,  # Increased for better tracking
        n_init=5,    # More confirmations required
        max_cosine_distance=0.2,  # Tighter matching
        nms_max_overlap=0.8
    )
    return detector, tracker

def detect_and_track_fast(frame, detector, tracker, conf_thres=0.3):
    """
    Optimized detection and tracking
    """
    # Use smaller inference size for speed
    results = detector(frame, verbose=False, imgsz=320)[0]
    
    detections = []
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < conf_thres:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        cls_name = detector.names[cls_id]
        
        # Add detection with features for better tracking
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_name))
    
    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)
    return tracks

# ========================
#  OBJECT COORDINATES & MOTION VECTORS
# ========================
class ObjectTracker:
    """
    Enhanced tracking with motion vectors and coordinates
    """
    def __init__(self, max_history=50):
        self.track_history = {}  # track_id -> deque of (center_x, center_y)
        self.motion_vectors = {}  # track_id -> (dx, dy, speed)
        self.max_history = max_history
        self.inactive_counter = {}  # 🆕 Thêm dòng này!
    
    def update_track(self, track_id, bbox):
        """
        Update track history and calculate motion vector
        bbox: (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=self.max_history)
        
        history = self.track_history[track_id]
        history.append((center_x, center_y))
        
        # Calculate motion vector if we have enough history
        if len(history) >= 2:
            prev_x, prev_y = history[-2]
            curr_x, curr_y = history[-1]
            
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            speed = np.sqrt(dx**2 + dy**2)
            
            self.motion_vectors[track_id] = (dx, dy, speed)
        else:
            self.motion_vectors[track_id] = (0, 0, 0)
    
    def get_coordinates(self, track_id):
        """Get current coordinates for a track"""
        if track_id in self.track_history and self.track_history[track_id]:
            return self.track_history[track_id][-1]
        return None
    
    def get_motion_vector(self, track_id):
        """Get motion vector for a track"""
        return self.motion_vectors.get(track_id, (0, 0, 0))
    
    def get_trajectory(self, track_id, max_points=20):
        """Get trajectory points for drawing"""
        if track_id in self.track_history:
            return list(self.track_history[track_id])[-max_points:]
        return []
    
    def cleanup_old_tracks(self, active_track_ids, max_inactive_frames=30):
        """Remove tracks that are no longer active for X frames"""
        current_ids = set(active_track_ids)
        
        # Chỉ xóa các track không active trong nhiều frame
        for track_id in list(self.track_history.keys()):
            if track_id not in current_ids:
                # Tăng counter cho track không active
                if track_id not in self.inactive_counter:
                    self.inactive_counter[track_id] = 1
                else:
                    self.inactive_counter[track_id] += 1
                
                # Chỉ xóa sau nhiều frame không active
                if self.inactive_counter[track_id] > max_inactive_frames:
                    del self.track_history[track_id]
                    if track_id in self.motion_vectors:
                        del self.motion_vectors[track_id]
                    del self.inactive_counter[track_id]
            else:
                # Reset counter nếu track active lại
                if track_id in self.inactive_counter:
                    del self.inactive_counter[track_id]


def get_object_coordinates(tracks, object_tracker):
    """
    Extract coordinates for all tracked objects
    Returns: list of (track_id, center_x, center_y, bbox, motion_vector)
    """
    coordinates = []
    active_track_ids = []
    
    for track in tracks:
        if not track.is_confirmed():
            continue
            
        track_id = track.track_id
        active_track_ids.append(track_id)
        
        try:
            bbox = track.to_ltrb()  # [left, top, right, bottom]
            l, t, r, b = map(int, bbox)
            
            # Update tracker
            object_tracker.update_track(track_id, (l, t, r, b))
            
            # Get coordinates and motion
            center_coords = object_tracker.get_coordinates(track_id)
            motion_vector = object_tracker.get_motion_vector(track_id)
            
            if center_coords:
                center_x, center_y = center_coords
                coordinates.append({
                    'track_id': track_id,
                    'center_x': center_x,
                    'center_y': center_y,
                    'bbox': (l, t, r, b),
                    'motion_vector': motion_vector,  # (dx, dy, speed)
                    'class_name': getattr(track, 'det_class', 'object')
                })
        except Exception as e:
            print(f"⚠️ Error processing track {track_id}: {e}")
            continue
    
    # Clean up old tracks
    object_tracker.cleanup_old_tracks(active_track_ids)
    
    return coordinates

# ========================
#  DRAWING (ENHANCED)
# ========================
def draw_tracks_enhanced(frame, tracks, object_tracker=None):
    """
    Enhanced drawing with motion vectors and trajectories
    """
    for track in tracks:
        if not track.is_confirmed():
            continue
            
        track_id = track.track_id
        
        try:
            l, t, r, b = map(int, track.to_ltrb())
            label = getattr(track, 'det_class', 'object')
            
            # Draw bounding box
            color = (0, 255, 0)  # Green for confirmed tracks
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            
            # Draw ID and label
            cv2.putText(
                frame,
                f"{label}-{track_id}",
                (l, t - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            
            # Draw motion vector and trajectory if available
            if object_tracker:
                # Draw trajectory
                trajectory = object_tracker.get_trajectory(track_id)
                for i in range(1, len(trajectory)):
                    cv2.line(frame, 
                            (int(trajectory[i-1][0]), int(trajectory[i-1][1])),
                            (int(trajectory[i][0]), int(trajectory[i][1])),
                            (255, 0, 0), 2)
                
                # Draw motion vector
                motion_vec = object_tracker.get_motion_vector(track_id)
                dx, dy, speed = motion_vec
                center_coords = object_tracker.get_coordinates(track_id)
                
                if center_coords and speed > 0.1:  # Only draw if moving
                    center_x, center_y = center_coords
                    end_x = int(center_x + dx * 5)  # Scale for visibility
                    end_y = int(center_y + dy * 5)
                    cv2.arrowedLine(frame, 
                                  (int(center_x), int(center_y)),
                                  (end_x, end_y),
                                  (0, 0, 255), 2, tipLength=0.3)
                    
                    # Display speed
                    cv2.putText(frame, f"S:{speed:.1f}", 
                              (l, t - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        except Exception as e:
            print(f"⚠️ Error drawing track {track_id}: {e}")
            continue
    
    return frame

# ========================
#  LOGGING (ENHANCED)
# ========================
def init_logger(log_dir="logs", log_file="detections.log"):
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, log_file)

def log_detection_with_coords(log_path, obj_id, obj_type, coordinates, motion_vector):
    """Enhanced logging with coordinates and motion"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    dx, dy, speed = motion_vector
    line = (f"[{timestamp}] ID={obj_id} | type={obj_type} | "
           f"pos=({coordinates[0]:.1f}, {coordinates[1]:.1f}) | "
           f"motion=({dx:.2f}, {dy:.2f}) | speed={speed:.2f}")
    print(line)
    with open(log_path, "a") as f:
        f.write(line + "\n")

def save_crop(frame, bbox, obj_id, obj_type, save_dir="logs/crops"):
    os.makedirs(save_dir, exist_ok=True)
    l, t, r, b = map(int, bbox)
    # Add padding and ensure within frame bounds
    pad = 5
    l = max(0, l - pad)
    t = max(0, t - pad)
    r = min(frame.shape[1], r + pad)
    b = min(frame.shape[0], b + pad)
    
    crop = frame[t:b, l:r]
    if crop.size > 0:
        filename = f"{obj_type}_{obj_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        cv2.imwrite(os.path.join(save_dir, filename), crop)
