import argparse
from src.camera import CameraController


def main():
    parser = argparse.ArgumentParser(
        description="Webcam Hand Tracker - Sign Language Recognition Demonstrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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

