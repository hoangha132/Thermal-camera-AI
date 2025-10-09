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

def main(source="videos\people2.mp4", model_path="yolov8n.pt"):
    # Init models + logger
    detector, tracker = init_models(model_path)
    log_path = init_logger()

    prev_time = 0
    fps = 0

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
        # frame = preprocess_frame(frame, is_thermal=True, use_retinex=True)
        frame = cv2.resize(frame, (640, 480))

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

        # Draw FPS on frame
        cv2.putText(frame, f"FPS: {fps:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)
        
        cv2.imshow("Thermal Detection + Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
