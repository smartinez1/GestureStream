import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import os
from collections import deque
import time
import urllib.request


class GestureStreamController:
    def __init__(self, image_path_1=None, image_path_2=None):
        # MediaPipe API setup
        self.BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        # Download model if not exists
        model_path = self._get_model_path()

        # Create hand landmarker with VIDEO mode
        options = self.HandLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=model_path),
            running_mode=self.VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hand_landmarker = self.HandLandmarker.create_from_options(options)

        # Gesture state - Initialize FIRST
        self.gesture_active = False
        self.cube_rotation = np.array([0.0, 0.0, 0.0])
        self.gesture_hand_position = None
        self.control_hand_position = None
        self.all_hand_landmarks = []
        self.last_open_hand_time = 0
        self.rotation_history = deque(maxlen=5)

        # Load images
        self.image_1 = self._load_or_create_image(image_path_1, (200, 200), "gradient")
        self.image_2 = self._load_or_create_image(image_path_2, (200, 200), "checkerboard")
        self.current_image = self.image_1.copy()

    def _get_model_path(self):
        """Get or download the hand landmarker model."""
        model_dir = os.path.expanduser("~/.mediapipe/models")
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "hand_landmarker.task")

        if os.path.exists(model_path):
            print(f"Using cached model: {model_path}")
            return model_path

        print("Downloading hand landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"Model downloaded to: {model_path}")
            return model_path
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise

    def _load_or_create_image(self, path, size, style="gradient"):
        """Load image or create a test image."""
        if path is not None and os.path.exists(path):
            img = Image.open(path).convert("RGB")
            img = img.resize(size, Image.Resampling.LANCZOS)
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        else:
            # Create test images
            if style == "gradient":
                img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                for i in range(size[0]):
                    img[:, i] = [int(i / size[0] * 255), 100, 150]
                return img
            elif style == "checkerboard":
                img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                block_size = 20
                for i in range(0, size[0], block_size):
                    for j in range(0, size[1], block_size):
                        if ((i // block_size) + (j // block_size)) % 2 == 0:
                            img[j : j + block_size, i : i + block_size] = [50, 200, 100]
                        else:
                            img[j : j + block_size, i : i + block_size] = [200, 50, 100]
                return img

    def detect_pointing_gesture(self, landmarks):
        """Detect if hand is in pointing gesture (index finger extended)."""
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]

        # Pointing: index extended, other fingers curled
        index_extended = index_tip.y < index_pip.y - 0.05
        middle_curled = middle_tip.y > middle_pip.y - 0.02
        ring_curled = ring_tip.y > ring_pip.y - 0.02
        pinky_curled = pinky_tip.y > pinky_pip.y - 0.02

        return index_extended and middle_curled and ring_curled and pinky_curled

    def detect_open_hand_gesture(self, landmarks):
        """Detect if hand is open (all fingers extended)."""
        tips = [4, 8, 12, 16, 20]
        pips = [3, 6, 10, 14, 18]

        extended_count = 0
        for tip_idx, pip_idx in zip(tips, pips):
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            if tip.y < pip.y - 0.03:
                extended_count += 1

        return extended_count >= 4

    def calculate_hand_orientation(self, landmarks):
        """Calculate hand orientation from palm and finger positions."""
        wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
        middle_mcp = np.array([landmarks[9].x, landmarks[9].y, landmarks[9].z])

        forward = middle_mcp - wrist
        forward = forward / (np.linalg.norm(forward) + 1e-6)

        thumb = np.array([landmarks[4].x, landmarks[4].y, landmarks[4].z])
        right_vec = np.cross(forward, thumb - wrist)
        right_vec = right_vec / (np.linalg.norm(right_vec) + 1e-6)

        pitch = np.arctan2(forward[1], forward[2])
        yaw = np.arctan2(forward[0], forward[2])
        roll = np.arctan2(right_vec[1], right_vec[0])

        return np.array([pitch, yaw, roll])

    def draw_3d_cube(self, frame, position, rotation, size=50):
        """Draw a 3D cube on frame based on rotation."""
        h, w = frame.shape[:2]

        vertices = np.array(
            [
                [-size, -size, -size],
                [size, -size, -size],
                [size, size, -size],
                [-size, size, -size],
                [-size, -size, size],
                [size, -size, size],
                [size, size, size],
                [-size, size, size],
            ]
        )

        rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(rotation[0]), -np.sin(rotation[0])],
                [0, np.sin(rotation[0]), np.cos(rotation[0])],
            ]
        )
        ry = np.array(
            [
                [np.cos(rotation[1]), 0, np.sin(rotation[1])],
                [0, 1, 0],
                [-np.sin(rotation[1]), 0, np.cos(rotation[1])],
            ]
        )
        rz = np.array(
            [
                [np.cos(rotation[2]), -np.sin(rotation[2]), 0],
                [np.sin(rotation[2]), np.cos(rotation[2]), 0],
                [0, 0, 1],
            ]
        )

        rotation_matrix = rz @ ry @ rx
        rotated_vertices = vertices @ rotation_matrix.T

        projected = []
        for vertex in rotated_vertices:
            x = int(position[0] + vertex[0])
            y = int(position[1] + vertex[1])
            projected.append([x, y])

        projected = np.array(projected)

        edges = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]

        for edge in edges:
            pt1 = tuple(projected[edge[0]])
            pt2 = tuple(projected[edge[1]])
            if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

    def overlay_image_on_hand(self, frame, image, hand_position, scale=0.5):
        """Overlay image on hand position."""
        if hand_position is None:
            return frame

        h, w = frame.shape[:2]
        img_h, img_w = image.shape[:2]

        scaled_w = int(img_w * scale)
        scaled_h = int(img_h * scale)
        scaled_image = cv2.resize(image, (scaled_w, scaled_h))

        x = int(hand_position[0] - scaled_w / 2)
        y = int(hand_position[1] - scaled_h / 2)

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + scaled_w)
        y2 = min(h, y + scaled_h)

        img_x1 = max(0, -x)
        img_y1 = max(0, -y)
        img_x2 = img_x1 + (x2 - x1)
        img_y2 = img_y1 + (y2 - y1)

        if x2 > x1 and y2 > y1:
            frame[y1:y2, x1:x2] = cv2.addWeighted(
                frame[y1:y2, x1:x2], 0.5, scaled_image[img_y1:img_y2, img_x1:img_x2], 0.5, 0
            )

        return frame

    def draw_hand_landmarks(self, frame, landmarks):
        """Draw hand landmarks on frame."""
        h, w = frame.shape[:2]

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17),
        ]

        for connection in connections:
            start_idx, end_idx = connection
            start = landmarks[start_idx]
            end = landmarks[end_idx]

            start_pos = (int(start.x * w), int(start.y * h))
            end_pos = (int(end.x * w), int(end.y * h))

            cv2.line(frame, start_pos, end_pos, (0, 255, 0), 2)

        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

        return frame

    def process_frame(self, frame, frame_timestamp_ms):
        """Process frame for hand detection and gesture recognition."""
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        self.gesture_hand_position = None
        self.control_hand_position = None
        self.all_hand_landmarks = []

        if result.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
                landmarks = list(hand_landmarks)
                self.all_hand_landmarks.append(landmarks)

                hand_center_x = np.mean([lm.x for lm in landmarks]) * w
                hand_center_y = np.mean([lm.y for lm in landmarks]) * h
                hand_center = (hand_center_x, hand_center_y)

                if self.detect_pointing_gesture(landmarks):
                    self.gesture_hand_position = hand_center
                    orientation = self.calculate_hand_orientation(landmarks)
                    self.rotation_history.append(orientation)

                    if len(self.rotation_history) > 0:
                        self.cube_rotation = np.mean(
                            list(self.rotation_history), axis=0
                        )

                elif self.detect_open_hand_gesture(landmarks):
                    self.control_hand_position = hand_center
                    current_time = time.time()
                    if current_time - self.last_open_hand_time > 0.5:
                        self.gesture_active = not self.gesture_active
                        if self.gesture_active:
                            self.current_image = self.image_2.copy()
                        else:
                            self.current_image = self.image_1.copy()
                        self.last_open_hand_time = current_time

                frame = self.draw_hand_landmarks(frame, landmarks)

        return frame

    def run(self, video_source=0):
        """Run the gesture stream controller."""
        cap = cv2.VideoCapture(video_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("Gesture Stream Active!")
        print("- Point with one hand to control the cube")
        print("- Open your other hand to switch images")
        print("- Press 'q' to quit")

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_timestamp_ms = int((time.time() - start_time) * 1000)
            frame = self.process_frame(frame, frame_timestamp_ms)

            if self.gesture_hand_position is not None:
                self.draw_3d_cube(
                    frame,
                    (int(self.gesture_hand_position[0]), int(self.gesture_hand_position[1])),
                    self.cube_rotation,
                    size=40,
                )

            if self.control_hand_position is not None:
                frame = self.overlay_image_on_hand(
                    frame, self.current_image, self.control_hand_position, scale=0.3
                )

            status = "Image 2 Active" if self.gesture_active else "Image 1 Active"
            cv2.putText(
                frame,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30.0 / (time.time() - start_time) if frame_count > 0 else 0
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 0),
                    2,
                )

            cv2.imshow("Gesture Stream Controller", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    controller = GestureStreamController(
        image_path_1=None,
        image_path_2=None,
    )
    controller.run()


if __name__ == "__main__":
    main()
