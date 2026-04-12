import cv2
from pathlib import Path
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandTracker:

    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        import urllib.request
        import os

        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.landmark_names = [
            "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
            "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
            "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
            "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
            "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
        ]

        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print("Downloading MediaPipe hand landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("Model downloaded")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        print("Hand tracker initialized")

    def detect_hands(self, frame):

        from mediapipe import Image, ImageFormat

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        results = self.landmarker.detect(mp_image)
        return results

    def get_hand_count(self, results):

        if not results or not results.hand_landmarks:
            return 0
        return len(results.hand_landmarks)

    def get_hand_info(self, results, flip_hands=False):

        hand_info = []
        if not results or not results.hand_landmarks or not results.handedness:
            return hand_info

        num_hands = len(results.hand_landmarks)
        num_handedness = len(results.handedness)

        if num_hands != num_handedness:
            return hand_info

        for idx in range(num_hands):
            landmarks = results.hand_landmarks[idx]
            handedness = results.handedness[idx]

            if landmarks and handedness:
                hand_name = handedness[0].display_name
                if flip_hands:
                    hand_name = "Left" if hand_name == "Right" else "Right"

                info = {
                    'handedness': hand_name,
                    'confidence': handedness[0].score,
                    'num_landmarks': len(landmarks),
                    'wrist': (landmarks[0].x, landmarks[0].y, landmarks[0].z) if landmarks else (0, 0, 0)
                }
                hand_info.append(info)

        return hand_info

    def extract_landmarks(self, results):

        landmarks_data = []

        if not results or not results.hand_landmarks:
            return landmarks_data

        for hand_idx, hand_landmarks in enumerate(results.hand_landmarks):
            hand_data = []
            for landmark in hand_landmarks:
                hand_data.extend([landmark.x, landmark.y, landmark.z, landmark.visibility if landmark.visibility else 0])
            landmarks_data.append(hand_data)

        return landmarks_data

    def draw_landmarks(self, frame, results, show_labels=False, show_confidence=False):

        if not results or not results.hand_landmarks or len(results.hand_landmarks) == 0:
            return frame

        h, w, _ = frame.shape

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]

        num_hands = len(results.hand_landmarks)
        num_handedness = len(results.handedness) if results.handedness else 0

        for hand_idx in range(num_hands):
            hand_landmarks = results.hand_landmarks[hand_idx]

            if not hand_landmarks or len(hand_landmarks) == 0:
                continue

            hand_color = (0, 255, 0)
            if hand_idx < num_handedness and results.handedness:
                handedness_obj = results.handedness[hand_idx]
                if handedness_obj and len(handedness_obj) > 0:
                    is_right = handedness_obj[0].display_name == "Right"
                    hand_color = (0, 255, 0) if is_right else (255, 0, 0)

            for landmark_idx, landmark in enumerate(hand_landmarks):
                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

                if show_labels:
                    label = self.landmark_names[landmark_idx] if landmark_idx < len(self.landmark_names) else str(landmark_idx)
                    cv2.putText(frame, label, (x + 5, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

                if show_confidence and landmark.visibility is not None and landmark.visibility > 0:
                    confidence_text = f"{landmark.visibility:.2f}"
                    cv2.putText(frame, confidence_text, (x + 10, y + 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1)

            for start, end in connections:
                if start < len(hand_landmarks) and end < len(hand_landmarks):
                    start_landmark = hand_landmarks[start]
                    end_landmark = hand_landmarks[end]

                    if not start_landmark or not end_landmark:
                        continue

                    start_visibility = start_landmark.visibility if start_landmark.visibility is not None else 1.0
                    end_visibility = end_landmark.visibility if end_landmark.visibility is not None else 1.0

                    if start_visibility > 0.1 and end_visibility > 0.1:
                        start_pos = (int(start_landmark.x * w), int(start_landmark.y * h))
                        end_pos = (int(end_landmark.x * w), int(end_landmark.y * h))
                        cv2.line(frame, start_pos, end_pos, (0, 255, 0), 2)

        return frame

    def release(self):
        pass


class CameraController:

    def __init__(self, camera_index=0, max_num_hands=2,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 snapshot_dir="snapshots"):

        self.camera_index = camera_index
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(exist_ok=True)

        self.tracker = HandTracker(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera with index {camera_index}")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        self.frame_count = 0
        self.show_confidence = False
        self.show_labels = False

        print(f"Camera opened successfully")
        print(f"  Resolution: {self.frame_width}x{self.frame_height}")
        print(f"  FPS: {self.fps}")

    def save_snapshot(self, frame):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.snapshot_dir / f"snapshot_{timestamp}.png"
        cv2.imwrite(str(filename), frame)
        print(f"Snapshot saved: {filename}")

    def add_ui_elements(self, frame, results=None, flip_hands=True):
        y_pos = 20

        controls_text = "Controls: 'c' confidence | 'l' labels | 's' save | 'q' quit"
        cv2.putText(frame, controls_text, (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 25

        res_text = f"Resolution: {self.frame_width}x{self.frame_height} | FPS: {self.fps}"
        cv2.putText(frame, res_text, (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 25

        if results:
            hand_count = self.tracker.get_hand_count(results)
            hands_text = f"Hands detected: {hand_count}"
            cv2.putText(frame, hands_text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if hand_count > 0 else (0, 0, 255), 1)
            y_pos += 25

            if hand_count > 0:
                hand_info = self.tracker.get_hand_info(results, flip_hands=flip_hands)
                for hand_idx, info in enumerate(hand_info):
                    hand_text = f"Hand {hand_idx + 1}: {info['handedness']} ({info['confidence']:.2%})"
                    hand_color = (0, 255, 0) if info['handedness'] == 'Right' else (255, 0, 0)
                    cv2.putText(frame, hand_text, (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1)
                    y_pos += 25

        status_text = f"Confidence: {'ON' if self.show_confidence else 'OFF'} | Labels: {'ON' if self.show_labels else 'OFF'}"
        cv2.putText(frame, status_text, (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return frame

    def run(self):
        print("\n" + "="*60)
        print("Webcam Hand Tracker - Sign Language Recognition")
        print("="*60)
        print("Controls:")
        print("  'c' - Toggle confidence display")
        print("  'l' - Toggle landmark labels")
        print("  's' - Save snapshot")
        print("  'q' - Quit")
        print("="*60 + "\n")

        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret:
                print("Error: Cannot read from camera")
                break

            frame = cv2.flip(frame, 1)
            self.frame_count += 1

            results = self.tracker.detect_hands(frame)

            if results:
                frame = self.tracker.draw_landmarks(
                    frame, results,
                    show_labels=self.show_labels,
                    show_confidence=self.show_confidence
                )

            frame = self.add_ui_elements(frame, results, flip_hands=True)

            cv2.imshow("Hand Tracker - Gebarentaal naar Tekst", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting application...")
                break
            elif key == ord('s'):
                self.save_snapshot(frame)
            elif key == ord('c'):
                self.show_confidence = not self.show_confidence
                print(f"Confidence display: {'ON' if self.show_confidence else 'OFF'}")
            elif key == ord('l'):
                self.show_labels = not self.show_labels
                print(f"Landmark labels: {'ON' if self.show_labels else 'OFF'}")

        print(f"Total frames processed: {self.frame_count}")
        self.cleanup()

    def cleanup(self):
        self.cap.release()
        self.tracker.release()
        cv2.destroyAllWindows()
        print("Resources released")

