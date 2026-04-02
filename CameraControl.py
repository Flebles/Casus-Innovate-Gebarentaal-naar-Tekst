import argparse
import os
import time
import uuid
import mediapipe as mp
import numpy as np
import cv2 as cv

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
	if cv is None:
		raise RuntimeError("OpenCV is required to save snapshots.")
	os.makedirs(output_dir, exist_ok=True)
	filename = f"{uuid.uuid4().hex}.png"
	filepath = os.path.join(output_dir, filename)
	cv.imwrite(filepath, frame)
	return filepath


def draw_hand_center(frame: object, landmarks, frame_width: int, frame_height: int) -> None:
	if cv is None:
		raise RuntimeError("OpenCV is required to draw on frames.")
	if np is None:
		raise RuntimeError("NumPy is required to draw hand centers.")
	points = np.array([(lm.x * frame_width, lm.y * frame_height) for lm in landmarks.landmark])
	center = points.mean(axis=0).astype(int)
	cv.circle(frame, tuple(center), 6, (0, 255, 255), -1)  # type: ignore[arg-type]


def classify_hand_gesture(hand_landmarks, handedness: str | None = None) -> str:
	landmark = hand_landmarks.landmark

	index_extended = landmark[8].y < landmark[6].y
	middle_extended = landmark[12].y < landmark[10].y
	ring_extended = landmark[16].y < landmark[14].y
	pinky_extended = landmark[20].y < landmark[18].y

	if handedness == "Right":
		thumb_extended = landmark[4].x > landmark[3].x
	elif handedness == "Left":
		thumb_extended = landmark[4].x < landmark[3].x
	else:
		thumb_extended = abs(landmark[4].x - landmark[3].x) > 0.03

	extended_fingers = [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]
	count = sum(extended_fingers)

	if count == 0:
		return "Fist"
	if count == 5:
		return "Open Palm"
	if index_extended and not middle_extended and not ring_extended and not pinky_extended:
		return "Pointing"
	if index_extended and middle_extended and not ring_extended and not pinky_extended:
		return "Peace"
	return f"{count} Fingers"


def main() -> None:
	if cv is None:
		raise RuntimeError("OpenCV is not installed. Install requirements.txt first.")

	args = parse_args()
	mp_styles = mp.solutions.drawing_styles
	mp_drawing = mp.solutions.drawing_utils
	mp_hands = mp.solutions.hands

	cap = cv.VideoCapture(args.camera_index)
	if not cap.isOpened():
		raise RuntimeError("Could not open camera. Try another --camera-index.")

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

			frame = cv.flip(frame, 1)
			frame_height, frame_width = frame.shape[:2]

			rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
			rgb.flags.writeable = False
			results = hands.process(rgb)
			rgb.flags.writeable = True

			tracked_hands = 0
			if results.multi_hand_landmarks:
				for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
					tracked_hands += 1
					handedness = None
					if results.multi_handedness and idx < len(results.multi_handedness):
						handedness = results.multi_handedness[idx].classification[0].label
					gesture = classify_hand_gesture(hand_landmarks, handedness)
					mp_drawing.draw_landmarks(
						frame,
						hand_landmarks,
						mp_hands.HAND_CONNECTIONS,
						mp_styles.get_default_hand_landmarks_style(),
						mp_styles.get_default_hand_connections_style(),
					)
					draw_hand_center(frame, hand_landmarks, frame_width, frame_height)

					if handedness:
						label_pos = hand_landmarks.landmark[0]
						x = int(label_pos.x * frame_width)
						y = int(label_pos.y * frame_height) - 10
						cv.putText(
							frame,
							f"{handedness}: {gesture}",
							(x, max(20, y)),
										cv.FONT_HERSHEY_SIMPLEX,
							0.7,
							(255, 255, 255),
							2,
										cv.LINE_AA,
						)
					else:
						label_pos = hand_landmarks.landmark[0]
						x = int(label_pos.x * frame_width)
						y = int(label_pos.y * frame_height) - 10
						cv.putText(
							frame,
							gesture,
							(x, max(20, y)),
										cv.FONT_HERSHEY_SIMPLEX,
							0.7,
							(255, 255, 255),
							2,
										cv.LINE_AA,
						)

			current_time = time.time()
			fps = 1.0 / max(current_time - previous_time, 1e-6)
			previous_time = current_time

			cv.putText(frame, f"Hands: {tracked_hands}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2)
			cv.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2)
			cv.putText(
				frame,
				"Press 's' to save snapshot, 'q' to quit",
				(10, frame_height - 10),
				cv.FONT_HERSHEY_SIMPLEX,
				0.6,
				(240, 240, 240),
				2,
			)

			cv.imshow("Hand Tracker", frame)
			key = cv.waitKey(1) & 0xFF

			if key == ord("s"):
				saved = save_snapshot(frame, args.snapshot_dir)
				print(f"Snapshot saved: {saved}")
			elif key == ord("q"):
				break

	cap.release()
	cv.destroyAllWindows()


if __name__ == "__main__":
	main()
