import cv2
import time
import numpy as np
from collections import deque
from utils import (
    preprocess_frame_fast,
    preprocess_raw_thermal_fast,
    init_models,
    detect_and_track_fast,
    draw_tracks_enhanced,
    init_logger,
    log_detection_with_coords,
    save_crop,
    ObjectTracker,
    get_object_coordinates
)

def resize_with_padding(frame, target_size=(640, 480)):
    """
    Fast resize with padding
    """
    target_w, target_h = target_size
    h, w = frame.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    frame_padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return frame_padded

def main(source="videos/xe-tăng.mp4", model_path="yolov10s-ver2.pt"):
    try:
        # Initialize models
        detector, tracker = init_models(model_path, device="cpu")  # Use CPU for stability
        object_tracker = ObjectTracker(max_history=30)
        log_path = init_logger()

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("❌ Error: Cannot open source.")
            return

        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"📹 Source FPS: {original_fps}")

        # FPS calculation
        prev_time = time.time()
        fps_buffer = deque(maxlen=30)
        paused = False
        frame_skip = 1  # Start with no frame skipping
        frame_counter = 0
        
        # 🆕 THÊM: Biến đếm track duy nhất
        unique_track_ids = set()

        print("🚀 Starting detection... (Press 'p' to pause, 'q' to quit)")
        print("📊 Controls: '+'=skip frames, '-'=less skipping, 'd'=toggle debug")

        debug_mode = False

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("✅ End of video reached")
                    break

                frame_counter += 1
                if frame_counter % frame_skip != 0:
                    continue

                # Resize first for speed
                frame = resize_with_padding(frame, (640, 480))

                # Optional preprocessing (comment out if too slow)
                # frame = preprocess_frame_fast(frame, is_thermal=True)
                # frame = preprocess_raw_thermal_fast(frame)

                try:
                    # Detection and tracking
                    tracks = detect_and_track_fast(frame, detector, tracker, conf_thres=0.3)

                    # Get object coordinates and motion vectors
                    coordinates_list = get_object_coordinates(tracks, object_tracker)
                    
                    # 🆕 CẬP NHẬT: Đếm track ID duy nhất
                    for coords in coordinates_list:
                        track_id = coords['track_id']
                        unique_track_ids.add(track_id)
                    
                    # Log detections
                    if debug_mode:
                        for coords in coordinates_list:
                            log_detection_with_coords(
                                log_path, 
                                coords['track_id'], 
                                coords['class_name'],
                                (coords['center_x'], coords['center_y']),
                                coords['motion_vector']
                            )

                    # Enhanced drawing
                    frame = draw_tracks_enhanced(frame, tracks, object_tracker)

                except Exception as e:
                    print(f"❌ Processing error: {e}")
                    continue

                # Calculate smooth FPS
                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                fps_buffer.append(fps)
                smooth_fps = sum(fps_buffer) / len(fps_buffer) if fps_buffer else 0
                prev_time = curr_time

                # Display information
                cv2.putText(frame, f"FPS: {smooth_fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Objects: {len(coordinates_list)}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Skip: {frame_skip}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                if debug_mode:
                    cv2.putText(frame, "DEBUG", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            else:
                # When paused
                cv2.putText(frame, "⏸ PAUSED", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.imshow("Thermal Detection + Tracking", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                paused = not paused
                print(f"{'⏸ Paused' if paused else '▶ Resumed'}")
                time.sleep(0.2)
            elif key == ord("+"):  # Increase frame skip for speed
                frame_skip = min(frame_skip + 1, 5)
                print(f"🔧 Frame skip: {frame_skip}")
            elif key == ord("-"):  # Decrease frame skip for accuracy
                frame_skip = max(frame_skip - 1, 1)
                print(f"🔧 Frame skip: {frame_skip}")
            elif key == ord("d"):  # Toggle debug mode
                debug_mode = not debug_mode
                print(f"🔧 Debug mode: {'ON' if debug_mode else 'OFF'}")

    except Exception as e:
        print(f"❌ Fatal error: {e}")
    
    finally:
        # Cleanup
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        
        # Performance summary
        if 'fps_buffer' in locals() and fps_buffer:
            avg_fps = sum(fps_buffer) / len(fps_buffer)
            print(f"📊 Average FPS: {avg_fps:.1f}")
        
        # 🆕 SỬA: Sử dụng unique_track_ids thay vì object_tracker.track_history
        print(f"📈 Total unique tracks detected: {len(unique_track_ids)}")
        
        # Optional: Vẫn hiển thị track hiện tại trong memory để debug
        # if 'object_tracker' in locals():
        #     print(f"📉 Current active tracks in memory: {len(object_tracker.track_history)}")

if __name__ == "__main__":
    main()
