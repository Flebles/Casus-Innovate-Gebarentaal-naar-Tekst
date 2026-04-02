import sys

from CameraControl import parse_args


def main() -> None:
    original_argv = sys.argv[:]
    try:
        sys.argv = ["CameraControl.py"]
        args = parse_args()
        assert args.camera_index == 0
        assert args.max_num_hands == 2
        assert args.min_detection == 0.6
        assert args.min_tracking == 0.5
        print("smoke_test.py passed")
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
