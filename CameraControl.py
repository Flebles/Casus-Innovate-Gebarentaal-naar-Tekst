import argparse
import datetime as dt
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Webcam hand tracker (OpenCV + MediaPipe)."
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index.")
    parser.add_argument(
        "--max-num-hands",
        type=int,
        default=1,
        help="Maximum number of hands MediaPipe should track.",
    )
    parser.add_argument(
        "--min-detection",
        type=float,
        default=0.6,
        help="Minimum hand detection confidence.",
    )
    parser.add_argument(
        "--min-tracking",
        type=float,
        default=0.5,
        help="Minimum hand tracking confidence.",
    )
    parser.add_argument(
        "--snapshot-dir",
        default="snapshots",
        help="Directory to save snapshots (key 's').",
    )
    return parser.parse_args()


def save_snapshot(frame, snapshot_dir: str) -> str:
    import cv2

    os.makedirs(snapshot_dir, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(snapshot_dir, f"snapshot_{timestamp}.png")
    cv2.imwrite(file_path, frame)
    return file_path


def run_camera(args: argparse.Namespace) -> int:
    # check voor dependencies, voor Noah en Milan
    try:
        import cv2
        import mediapipe as mp
    except ImportError as e:
        print("Missing dependency:", e)
        print("Run: pip install -r requirements.txt")
        return 1

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"Could not open camera at index {args.camera_index}.")
        return 1
    
    # set resolution, want sommige cameras defaulten op lage resolutie 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    print("Camera started. Press 'q' to quit, 's' to save a snapshot.")

    # cleanup
    try:
        with mp_hands.Hands(
            max_num_hands=args.max_num_hands,
            min_detection_confidence=args.min_detection,
            min_tracking_confidence=args.min_tracking,
        ) as hands:
            while True:
                ...
    finally:
        cap.release()
        cv2.destroyAllWindows()

    with mp_hands.Hands(
        max_num_hands=args.max_num_hands,
        min_detection_confidence=args.min_detection,
        min_tracking_confidence=args.min_tracking,
    ) as hands:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read a frame from the camera.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    if results.multi_handedness and idx < len(results.multi_handedness):
                        hand_label = results.multi_handedness[idx].classification[0].label
                        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                        h, w, _ = frame.shape

                        # keep text from going off-screen
                        x = int(wrist.x * w)
                        y = max(20, int(wrist.y * h) - 10)

                        cv2.putText(
                            frame,
                            hand_label,
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )

            cv2.putText(
                frame,
                "q=quit, s=snapshot",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Webcam Hand Tracker", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            # snapshot feedback on screen
            if key == ord("s"):
                path = save_snapshot(frame, args.snapshot_dir)
                print(f"Snapshot saved: {path}")
                cv2.putText(
                    frame,
                    "Snapshot saved!",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

    cap.release()
    cv2.destroyAllWindows()
    return 0


def main() -> None:
    args = parse_args()
    exit_code = run_camera(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()


