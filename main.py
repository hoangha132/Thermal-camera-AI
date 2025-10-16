import cv2
import time
from utils import (
    preprocess_frame,
    init_models,
    detect_and_track,
    draw_tracks,
    init_logger,
    log_detection,
    save_crop,
)

def resize_with_padding(frame, target_size=(640, 480)):
    """
    Resize frame to fit inside target_size while keeping aspect ratio.
    Adds black padding (letterbox) if needed.
    """
    target_w, target_h = target_size
    h, w = frame.shape[:2]

    # Compute scaling factor to fit frame inside target
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize with preserved aspect ratio
    resized = cv2.resize(frame, (new_w, new_h))

    # Compute padding values
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)

    # Add padding (black borders)
    frame_padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    return frame_padded


def main(source="videos\people.mp4", model_path="thermal1.pt"):
    # Init models + logger
    detector, tracker = init_models(model_path)
    log_path = init_logger()

    prev_time = 0
    fps = 0
    total_fps = 0
    frame_count = 0
    start_time = time.time()

    # Capture source (camera / RTSP / file)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("❌ Error: Cannot open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        frame = preprocess_frame(frame, is_thermal=True, use_retinex=True)
        frame = resize_with_padding(frame, (640, 480))

        # Detection + Tracking
        tracks = detect_and_track(frame, detector, tracker)

        # Handle each tracked object
        for t in tracks:
            if not t.is_confirmed():
                continue
            obj_id = t.track_id
            l, t_, r, b = map(int, t.to_ltrb())
            obj_type = t.get_det_class() if hasattr(t, "get_det_class") else "object"

            log_detection(log_path, obj_id, obj_type)
            save_crop(frame, (l, t_, r, b), obj_id, obj_type)

        # Draw results
        frame = draw_tracks(frame, tracks)

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        # Accumulate FPS for average calculation
        if fps > 0:
            total_fps += fps
            frame_count += 1

        # Draw FPS on frame
        cv2.putText(frame, f"FPS: {fps:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)
        
        cv2.imshow("Thermal Detection + Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

     # Calculate average FPS
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    print(f"\n📊 Average FPS across session: {avg_fps:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
