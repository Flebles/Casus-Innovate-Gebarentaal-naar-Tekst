import argparse
import os
import time
import uuid
import cv2
import mediapipe as mp
import numpy as np


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Real-time hand tracker with MediaPipe + OpenCV")
	parser.add_argument("--camera-index", type=int, default=0, help="Webcam index (default: 0)")
	parser.add_argument("--max-num-hands", type=int, default=2, help="Maximum number of tracked hands")
	parser.add_argument("--min-detection", type=float, default=0.6, help="Minimum detection confidence")
	parser.add_argument("--min-tracking", type=float, default=0.5, help="Minimum tracking confidence")
	parser.add_argument(
		"--snapshot-dir",
		type=str,
		default="snapshots",
		help="Directory for snapshots when pressing 's'",
	)
	return parser.parse_args()


def save_snapshot(frame: np.ndarray, output_dir: str) -> str:
	os.makedirs(output_dir, exist_ok=True)
	filename = f"{uuid.uuid4().hex}.png"
	filepath = os.path.join(output_dir, filename)
	cv2.imwrite(filepath, frame)
	return filepath


def draw_hand_center(frame: np.ndarray, landmarks, frame_width: int, frame_height: int) -> None:
	points = np.array([(lm.x * frame_width, lm.y * frame_height) for lm in landmarks.landmark])
	center = points.mean(axis=0).astype(int)
	cv2.circle(frame, tuple(center), 6, (0, 255, 255), -1)


def main() -> None:
	args = parse_args()

	cap = cv2.VideoCapture(args.camera_index)
	if not cap.isOpened():
		raise RuntimeError("Could not open camera. Try another --camera-index.")

	mp_hands = mp.solutions.hands
	mp_drawing = mp.solutions.drawing_utils
	mp_styles = mp.solutions.drawing_styles

	previous_time = time.time()

	with mp_hands.Hands(
		model_complexity=1,
		max_num_hands=args.max_num_hands,
		min_detection_confidence=args.min_detection,
		min_tracking_confidence=args.min_tracking,
	) as hands:
		while True:
			ok, frame = cap.read()
			if not ok:
				print("Camera frame could not be read.")
				break

			frame = cv2.flip(frame, 1)
			frame_height, frame_width = frame.shape[:2]

			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			rgb.flags.writeable = False
			results = hands.process(rgb)
			rgb.flags.writeable = True

			tracked_hands = 0
			if results.multi_hand_landmarks:
				for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
					tracked_hands += 1
					mp_drawing.draw_landmarks(
						frame,
						hand_landmarks,
						mp_hands.HAND_CONNECTIONS,
						mp_styles.get_default_hand_landmarks_style(),
						mp_styles.get_default_hand_connections_style(),
					)
					draw_hand_center(frame, hand_landmarks, frame_width, frame_height)

					if results.multi_handedness and idx < len(results.multi_handedness):
						handedness = results.multi_handedness[idx].classification[0].label
						label_pos = hand_landmarks.landmark[0]
						x = int(label_pos.x * frame_width)
						y = int(label_pos.y * frame_height) - 10
						cv2.putText(
							frame,
							handedness,
							(x, max(20, y)),
							cv2.FONT_HERSHEY_SIMPLEX,
							0.7,
							(255, 255, 255),
							2,
							cv2.LINE_AA,
						)

			current_time = time.time()
			fps = 1.0 / max(current_time - previous_time, 1e-6)
			previous_time = current_time

			cv2.putText(frame, f"Hands: {tracked_hands}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2)
			cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2)
			cv2.putText(
				frame,
				"Press 's' to save snapshot, 'q' to quit",
				(10, frame_height - 10),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.6,
				(240, 240, 240),
				2,
			)

			cv2.imshow("Hand Tracker", frame)
			key = cv2.waitKey(1) & 0xFF

			if key == ord("s"):
				saved = save_snapshot(frame, args.snapshot_dir)
				print(f"Snapshot saved: {saved}")
			elif key == ord("q"):
				break

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
