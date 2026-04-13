#!/usr/bin/env python3

import cv2
import argparse
from src.camera import HandTracker
from src.gesture import GestureDataManager


# Collect hand landmark data for a specific gesture
def collect_gesture_data(
    gesture_name: str,
    num_samples: int = 50,
    dataset_path: str = "data/gestures.csv",
    camera_index: int = 0
):
    # Initialize hand tracker and data manager
    tracker = HandTracker(max_num_hands=2, min_detection_confidence=0.5)
    data_manager = GestureDataManager(dataset_path)
    data_manager.initialize_dataset()

    # Open camera feed
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_index}")
        return

    print("\n" + "="*60)
    print(f"Collecting gesture: {gesture_name}")
    print("="*60)
    print(f"Target samples: {num_samples}")
    print("\nControls:")
    print("  SPACE - Record/Pause")
    print("  r     - Reset counter")
    print("  q     - Quit")
    print("="*60 + "\n")

    recording = False
    sample_count = 0

    # Main collection loop
    while sample_count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read from camera")
            break

        frame = cv2.flip(frame, 1)

        # Detect hand landmarks in frame
        results = tracker.detect_hands(frame)

        # Draw hand landmarks if detected
        if results and results.hand_landmarks:
            frame = tracker.draw_landmarks(
                frame,
                results,
                show_labels=True,
                show_confidence=True
            )

        status_color = (0, 255, 0) if recording else (100, 100, 100)
        cv2.putText(
            frame,
            f"Gesture: {gesture_name} | Progress: {sample_count}/{num_samples}",
            (15, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2
        )

        if recording:
            cv2.putText(
                frame,
                "RECORDING",
                (15, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
        else:
            cv2.putText(
                frame,
                "PAUSED (press SPACE to record)",
                (15, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 100, 100),
                1
            )

        cv2.putText(
            frame,
            "SPACE=Record | r=Reset | q=Quit",
            (15, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 150, 150),
            1
        )

        cv2.imshow("NGT Gesture Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            # Toggle recording on/off
            recording = not recording
            if recording:
                print(f"Recording {gesture_name}...")
            else:
                print(f"Paused. Samples: {sample_count}/{num_samples}")

        elif key == ord('r'):
            sample_count = 0
            recording = False
            print(f"Reset counter to 0")

        elif key == ord('q'):
            print(f"\nExiting... (collected {sample_count} samples)")
            break

        if recording and results and results.hand_landmarks:
            # Extract and store hand landmarks
            landmarks = tracker.extract_landmarks(results)

            flat_landmarks = []
            for hand_data in landmarks:
                flat_landmarks.extend(hand_data)

            while len(flat_landmarks) < 168:
                flat_landmarks.append(0.0)

            flat_landmarks = flat_landmarks[:168]

            data_manager.add_gesture_sample(gesture_name, flat_landmarks)
            sample_count += 1

            if sample_count % 10 == 0:
                print(f"{sample_count}/{num_samples} samples")

            if sample_count >= num_samples:
                break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nCollection complete!")
    print(f"   Gesture: {gesture_name}")
    print(f"   Samples: {sample_count}")
    print(f"   Saved to: {dataset_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Collect NGT gesture data for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Collect 50 samples of "1"
  python main_collect.py --gesture 1 --samples 50
  
  Collect to custom location
  python main_collect.py --gesture 2 --samples 50 --dataset data/ngt_data.csv
  
  Use different camera
  python main_collect.py --gesture 3 --samples 50 --camera 1
        """
    )

    parser.add_argument(
        "--gesture",
        type=str,
        required=True,
        help="Name of the gesture (e.g., 'hallo', 'dank_je', 'ja')"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples to collect (default: 50)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/gestures.csv",
        help="Output CSV path (default: data/gestures.csv)"
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)"
    )

    args = parser.parse_args()

    try:
        collect_gesture_data(
            gesture_name=args.gesture,
            num_samples=args.samples,
            dataset_path=args.dataset,
            camera_index=args.camera
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

