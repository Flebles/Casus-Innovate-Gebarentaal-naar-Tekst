#!/usr/bin/env python3
"""
Real-time Gesture Recognition

Loads a trained model and performs live gesture recognition from camera feed.

Usage:
    python main_recognize.py --model models/ngt_gesture_model.pkl --threshold 0.7
"""

import cv2
import argparse
from src.camera import HandTracker
from src.gesture import GestureClassifier


def recognize_gestures(model_path: str = "models/gesture_model.pkl", confidence_threshold: float = 0.5, camera_index: int = 0):
    """
    Real-time gesture recognition from camera feed.

    Args:
        model_path: Path to trained model
        confidence_threshold: Minimum confidence to display result
        camera_index: Camera index to use
    """

    tracker = HandTracker(max_num_hands=2)
    classifier = GestureClassifier(model_path)

    if not classifier.is_ready():
        print("Error: Model not ready. Please train a model first.")
        return

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_index}")
        return

    print("\n" + "="*60)
    print("Real-time Gesture Recognition")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Threshold: {confidence_threshold:.2f}")
    print("\nControls:")
    print("  '+' / '-' : Adjust confidence threshold")
    print("  's'       : Save snapshot")
    print("  'q'       : Quit")
    print("="*60 + "\n")

    current_threshold = confidence_threshold
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1
        h, w, _ = frame.shape

        results = tracker.detect_hands(frame)

        # Draw landmarks
        if results and results.hand_landmarks:
            frame = tracker.draw_landmarks(frame, results, show_labels=True)

        # Display threshold
        y_pos = 30
        cv2.putText(frame, f"Threshold: {current_threshold:.2f}", (15, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        y_pos += 35

        # Perform recognition
        if results and results.hand_landmarks:
            landmarks = tracker.extract_landmarks(results)

            # Flatten and pad landmarks for prediction
            flat_landmarks = []
            for hand_data in landmarks:
                flat_landmarks.extend(hand_data)

            # Pad to 168 features if needed
            while len(flat_landmarks) < 168:
                flat_landmarks.append(0.0)
            flat_landmarks = flat_landmarks[:168]

            gesture, confidence = classifier.predict(flat_landmarks)

            color = (0, 255, 0) if confidence >= current_threshold else (100, 100, 100)

            cv2.putText(frame, f"{gesture}", (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            y_pos += 40

            cv2.putText(frame, f"Confidence: {confidence:.1%}", (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)

        else:
            cv2.putText(frame, "No hands detected", (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 1)

        # Show frame counter
        cv2.putText(frame, f"Frame: {frame_count}", (15, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("Number Gesture Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n⏹ Exiting...")
            break

        elif key == ord('+') or key == ord('='):
            current_threshold = min(1.0, current_threshold + 0.05)
            print(f"✓ Threshold: {current_threshold:.2f}")

        elif key == ord('-') or key == ord('_'):
            current_threshold = max(0.0, current_threshold - 0.05)
            print(f"✓ Threshold: {current_threshold:.2f}")

        elif key == ord('s'):
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshots/recognition_{timestamp}.png"
            cv2.imwrite(filename, frame)
            print(f"💾 Snapshot saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time number gesture recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_recognize.py --model models/ngt_gesture_model.pkl
  python main_recognize.py --model models/ngt_gesture_model.pkl --threshold 0.75
  python main_recognize.py --model models/ngt_gesture_model.pkl --camera 1
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold 0.0-1.0 (default: 0.7)"
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)"
    )

    args = parser.parse_args()

    if not (0.0 <= args.threshold <= 1.0):
        print("Error: Threshold must be between 0.0 and 1.0")
        return 1

    try:
        recognize_gestures(args.model, args.threshold, args.camera)
        return 0
    except KeyboardInterrupt:
        print("\n⏹ Interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

