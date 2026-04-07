import cv2
import argparse
import sys
from pathlib import Path
from datetime import datetime

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandTracker:
    """Simple hand tracking using MediaPipe modern tasks API."""

    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """Initialize hand tracker."""
        import urllib.request
        import os

        # Download model if not exists
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print("Downloading MediaPipe hand landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("✓ Model downloaded")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        print("✓ Hand tracker initialized")

    def detect_hands(self, frame):
        """Detect hands in frame."""
        from mediapipe import Image, ImageFormat
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        results = self.landmarker.detect(mp_image)
        return results

    def draw_landmarks(self, frame, results):
        """Draw hand landmarks on frame."""
        if not results or not results.hand_landmarks:
            return frame

        h, w, _ = frame.shape

        for hand_landmarks in results.hand_landmarks:
            # Draw landmarks
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

            # Draw connections
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20)
            ]
            for start, end in connections:
                if start < len(hand_landmarks) and end < len(hand_landmarks):
                    start_pos = (int(hand_landmarks[start].x * w),
                               int(hand_landmarks[start].y * h))
                    end_pos = (int(hand_landmarks[end].x * w),
                             int(hand_landmarks[end].y * h))
                    cv2.line(frame, start_pos, end_pos, (0, 255, 0), 2)

        return frame

    def release(self):
        """Release resources."""
        pass


class CameraController:
    """Main application controller for webcam hand tracking."""

    def __init__(self, camera_index=0, max_num_hands=2,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 snapshot_dir="snapshots"):
        """
        Initialize the camera controller.

        Args:
            camera_index: Index of camera to use
            max_num_hands: Maximum hands to detect
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence
            snapshot_dir: Directory for saving snapshots
        """
        self.camera_index = camera_index
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(exist_ok=True)

        # Initialize hand tracker
        self.tracker = HandTracker(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Initialize webcam
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera with index {camera_index}")

        # Get camera properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        print(f"✓ Camera opened successfully")
        print(f"  Resolution: {self.frame_width}x{self.frame_height}")
        print(f"  FPS: {self.fps}")

    def save_snapshot(self, frame):
        """
        Save a snapshot of the current frame.

        Args:
            frame: Frame to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.snapshot_dir / f"snapshot_{timestamp}.png"
        cv2.imwrite(str(filename), frame)
        print(f"✓ Snapshot saved: {filename}")

    def add_ui_elements(self, frame):
        """
        Add UI elements to the frame.

        Args:
            frame: Frame to add UI to

        Returns:
            frame: Frame with UI elements
        """
        info_text = [
            "Controls: 's' to save | 'q' to quit",
            f"Resolution: {self.frame_width}x{self.frame_height} | FPS: {self.fps}"
        ]

        for i, text in enumerate(info_text):
            y_pos = 30 + (i * 25)
            cv2.putText(frame, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return frame

    def run(self):
        """Main run loop for the application."""
        print("\n" + "="*60)
        print("Webcam Hand Tracker - Sign Language Recognition")
        print("="*60)
        print("Controls:")
        print("  's' - Save snapshot")
        print("  'q' - Quit")
        print("="*60 + "\n")

        frame_count = 0

        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("Error: Cannot read from camera")
                break

            frame = cv2.flip(frame, 1)
            frame_count += 1

            results = self.tracker.detect_hands(frame)

            if results:
                frame = self.tracker.draw_landmarks(frame, results)

            frame = self.add_ui_elements(frame)

            cv2.imshow("Hand Tracker - Gebarentaal naar Tekst", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting application...")
                break
            elif key == ord('s'):
                self.save_snapshot(frame)

        print(f"Total frames processed: {frame_count}")
        self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.cap.release()
        self.tracker.release()
        cv2.destroyAllWindows()
        print("✓ Resources released")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Webcam Hand Tracker - Sign Language Recognition Demonstrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python CameraControls.py
  python CameraControls.py --camera-index 0 --max-num-hands 1
  python CameraControls.py --min-detection 0.7 --min-tracking 0.6
        """
    )

    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Index of camera to use (default: 0)"
    )

    parser.add_argument(
        "--max-num-hands",
        type=int,
        default=2,
        choices=[1, 2],
        help="Maximum number of hands to detect (default: 2)"
    )

    parser.add_argument(
        "--min-detection",
        type=float,
        default=0.5,
        help="Minimum detection confidence 0.0-1.0 (default: 0.5)"
    )

    parser.add_argument(
        "--min-tracking",
        type=float,
        default=0.5,
        help="Minimum tracking confidence 0.0-1.0 (default: 0.5)"
    )

    parser.add_argument(
        "--snapshot-dir",
        type=str,
        default="snapshots",
        help="Directory to save snapshots (default: snapshots)"
    )

    args = parser.parse_args()

    if not (0.0 <= args.min_detection <= 1.0):
        parser.error("--min-detection must be between 0.0 and 1.0")

    if not (0.0 <= args.min_tracking <= 1.0):
        parser.error("--min-tracking must be between 0.0 and 1.0")

    try:
        controller = CameraController(
            camera_index=args.camera_index,
            max_num_hands=args.max_num_hands,
            min_detection_confidence=args.min_detection,
            min_tracking_confidence=args.min_tracking,
            snapshot_dir=args.snapshot_dir
        )
        controller.run()

    except RuntimeError as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Check camera permissions in Windows Settings")
        print("  2. Ensure no other application is using the camera")
        print("  3. Try a different camera index: --camera-index 1")
        return 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1

    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())