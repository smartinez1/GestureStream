import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import glob
from collections import deque
import time
import urllib.request
from gesture_stream_diffusion import VoxelDiffusion


class GestureStreamController:
    def __init__(self, image_path_1=None, image_path_2=None, image_folder="images"):
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
        self.cube_rotation = np.array([0.0, 0.0, 0.0])
        self.projection_rotation = np.zeros(3)
        self.ORIENTATION_SMOOTHING = 0.03
        self.MAX_PROJECTION_TILT = 0.6
        self.gesture_hand_position = None
        self.control_hand_position = None
        self.all_hand_landmarks = []
        self.rotation_history = deque(maxlen=5)

        # Other-hand state (open/closed) and diffusion burst
        self.other_hand_open = False
        self.diffusion = VoxelDiffusion()
        self.openness_closed = deque(maxlen=90)
        self.openness_open = deque(maxlen=90)
        self.last_openness = 0.0

        # Toggle for hiding annotations (cubes, arrow, hand keypoints)
        self.show_annotations = True

        # Load images (folder takes priority, falls back to generated pair)
        self.images = self._load_images_from_folder(image_folder)
        if not self.images:
            self.image_1 = self._load_or_create_image(
                image_path_1, (200, 200), "gradient"
            )
            self.image_2 = self._load_or_create_image(
                image_path_2, (200, 200), "checkerboard"
            )
            self.images = [self.image_1.copy(), self.image_2.copy()]
        else:
            self.image_1 = self.images[0].copy()
            self.image_2 = (
                self.images[1].copy()
                if len(self.images) > 1
                else self.images[0].copy()
            )
        self.image_index = 0
        self.current_image = self.images[0].copy()

    def _composite_image(self, image_bgra):
        """Composite a BGRA image on white for palette extraction."""
        if image_bgra.shape[2] == 3:
            return image_bgra
        bgr = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2BGR)
        alpha = image_bgra[:, :, 3:4].astype(np.float32) / 255.0
        white = np.full_like(bgr, 255, dtype=np.float32)
        return (bgr.astype(np.float32) * alpha + white * (1.0 - alpha)).astype(
            np.uint8
        )

    def _ease_projection_rotation(self, target):
        """Very slowly ease the image-projection orientation toward the
        hand's orientation (which is jittery). Shortest-arc per axis so
        angles don't spin around when crossing the -pi/pi boundary.
        Pitch/yaw are clamped so the projected plane never tilts
        edge-on or back-facing — the image stays visible."""
        delta = (target - self.projection_rotation + np.pi) % (2 * np.pi) - np.pi
        eased = self.projection_rotation + delta * self.ORIENTATION_SMOOTHING
        eased[:2] = np.clip(
            eased[:2], -self.MAX_PROJECTION_TILT, self.MAX_PROJECTION_TILT
        )
        self.projection_rotation = eased

    def draw_image_on_cube(self, frame, position, size=50, rotation=None):
        """Draw the current image (with alpha) at the cube position.

        With `rotation`, the image is projected onto the plane
        perpendicular to the cube's orientation (same rotation as the
        cube, orthographic projection) instead of being axis-aligned."""
        h, w = frame.shape[:2]
        side = max(int(size * 4), 24)
        img = self.current_image

        if rotation is None:
            scaled = cv2.resize(img, (side, side))
            x = int(position[0] - side / 2)
            y = int(position[1] - side / 2)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + side), min(h, y + side)
            ix1, iy1 = max(0, -x), max(0, -y)
            ix2 = ix1 + (x2 - x1)
            iy2 = iy1 + (y2 - y1)
            if x2 <= x1 or y2 <= y1:
                return frame
            self._blend_image_into_frame(frame, scaled[iy1:iy2, ix1:ix2], x1, y1)
            return frame

        # 3D plane perpendicular to the orientation: rotate the image-plane
        # corners by the smoothed rotation, then orthographically project.
        half = side / 2.0
        rx, ry, rz = rotation
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        rot = rot_z @ rot_y @ rot_x

        corners = np.array(
            [
                [-half, -half, 0.0],
                [half, -half, 0.0],
                [half, half, 0.0],
                [-half, half, 0.0],
            ]
        )
        projected = (rot @ corners.T).T[:, :2]
        dst = projected + np.array([position[0], position[1]])

        x0 = int(np.floor(dst[:, 0].min()))
        y0 = int(np.floor(dst[:, 1].min()))
        x1 = int(np.ceil(dst[:, 0].max()))
        y1 = int(np.ceil(dst[:, 1].max()))
        if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
            return frame
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return frame

        src = np.float32(
            [
                [0, 0],
                [img.shape[1], 0],
                [img.shape[1], img.shape[0]],
                [0, img.shape[0]],
            ]
        )
        dst_local = dst - np.array([x0, y0])
        matrix = cv2.getPerspectiveTransform(src, dst_local.astype(np.float32))
        warped = cv2.warpPerspective(
            img,
            matrix,
            (x1 - x0, y1 - y0),
            flags=cv2.INTER_LINEAR,
            borderValue=(0, 0, 0, 0),
        )
        self._blend_image_into_frame(frame, warped, x0, y0)
        return frame

    def _blend_image_into_frame(self, frame, roi_img, x, y):
        """Alpha-composite a (possibly 4-channel) image region into frame."""
        h, w = frame.shape[:2]
        x2, y2 = x + roi_img.shape[1], y + roi_img.shape[0]
        if roi_img.shape[2] == 4:
            bgr = roi_img[:, :, :3].astype(np.float32)
            alpha = roi_img[:, :, 3:4].astype(np.float32) / 255.0
            roi = frame[y:y2, x:x2].astype(np.float32)
            frame[y:y2, x:x2] = (bgr * alpha + roi * (1.0 - alpha)).astype(
                np.uint8
            )
        else:
            frame[y:y2, x:x2] = roi_img

    def _hand_openness(self, landmarks):
        """Continuous openness of a hand: mean tip-to-PIP distance
        relative to palm size (scale invariant)."""
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        palm = max(
            np.linalg.norm(
                [wrist.x - middle_mcp.x, wrist.y - middle_mcp.y]
            ),
            1e-6,
        )
        distances = []
        for tip_idx, pip_idx in ((4, 3), (8, 6), (12, 10), (16, 14), (20, 18)):
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            distances.append(
                np.linalg.norm([tip.x - pip.x, tip.y - pip.y])
            )
        return float(np.mean(distances) / palm)

    def _transition_duration(self, history, current_openness, opening):
        """Estimate how long the hand took to leave its previous resting
        level and reach its new state (doubled for the animation, clamped
        to 0.8-6.0 s). `history`
        holds (timestamp, openness) samples from the previous state."""
        if not history:
            return VoxelDiffusion.DURATION
        prev_level = float(np.median([o for _, o in history]))
        span = abs(current_openness - prev_level)
        if span < 1e-3:
            return VoxelDiffusion.DURATION
        if opening:
            leave_level = prev_level + 0.15 * span
        else:
            leave_level = prev_level - 0.15 * span
        start_time = None
        for t, o in reversed(history):
            if (o <= leave_level) if opening else (o >= leave_level):
                start_time = t
                break
        if start_time is None:
            return VoxelDiffusion.DURATION
        return float(np.clip((time.time() - start_time) * 2.0, 0.8, 6.0))

    def _load_images_from_folder(self, folder):
        """Load all PNG/JPG images from a folder, sorted by name."""
        if not folder or not os.path.isdir(folder):
            return []
        paths = []
        for pattern in ("*.png", "*.jpg", "*.jpeg"):
            paths.extend(glob.glob(os.path.join(folder, pattern)))
        paths.sort()

        images = []
        for path in paths:
            img = Image.open(path)
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            img = img.resize((400, 400), Image.Resampling.LANCZOS)
            images.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA))
        return images

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
        if not self.show_annotations:
            return frame
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
        open_hand_detected = False

        if result.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
                landmarks = list(hand_landmarks)
                self.all_hand_landmarks.append(landmarks)

                hand_center_x = np.mean([lm.x for lm in landmarks]) * w
                hand_center_y = np.mean([lm.y for lm in landmarks]) * h
                hand_center = (hand_center_x, hand_center_y)

                is_pointing = self.detect_pointing_gesture(landmarks)
                is_open = self.detect_open_hand_gesture(landmarks)
                if is_pointing:
                    self.gesture_hand_position = hand_center
                    orientation = self.calculate_hand_orientation(landmarks)
                    self.rotation_history.append(orientation)
                    self._ease_projection_rotation(orientation)

                    if len(self.rotation_history) > 0:
                        self.cube_rotation = np.mean(
                            list(self.rotation_history), axis=0
                        )

                elif is_open:
                    self.control_hand_position = hand_center
                    open_hand_detected = True

                if not is_pointing:
                    # track openness per state for transition-speed
                    self.last_openness = self._hand_openness(landmarks)
                    history = (
                        self.openness_open if is_open else self.openness_closed
                    )
                    history.append((time.time(), self.last_openness))

                frame = self.draw_hand_landmarks(frame, landmarks)

        # Other hand state: explode on closed -> open, reconstruct the next
        # image (with morphing voxel colors) on open -> closed. Animation
        # duration follows the hand's transition speed.
        if open_hand_detected and not self.other_hand_open:
            if self.gesture_hand_position is not None:
                duration = self._transition_duration(
                    self.openness_closed, self.last_openness, opening=True
                )
                self.diffusion.trigger(
                    self._composite_image(self.current_image),
                    duration=duration,
                )
                print(
                    f"Diffusion: exploding image "
                    f"{self.image_index + 1}/{len(self.images)} "
                    f"({duration:.1f}s)"
                )
        elif not open_hand_detected and self.other_hand_open:
            if self.gesture_hand_position is not None:
                duration = self._transition_duration(
                    self.openness_open, self.last_openness, opening=False
                )
                self.image_index = (self.image_index + 1) % len(self.images)
                self.current_image = self.images[self.image_index].copy()
                self.diffusion.reconstruct(
                    self._composite_image(self.current_image),
                    duration=duration,
                )
                print(
                    f"Diffusion: reconstructing image "
                    f"{self.image_index + 1}/{len(self.images)} "
                    f"({duration:.1f}s)"
                )
        self.other_hand_open = open_hand_detected

        return frame

    def run(self, video_source=0):
        """Run the gesture stream controller."""
        cap = cv2.VideoCapture(video_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("Gesture Stream Active!")
        print("- Point with one hand to control the cube")
        print("- Open your other hand to project the next image on the cube")
        print("- 'h' to toggle annotations (cube, hand keypoints)")
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
                cube_pos = (
                    int(self.gesture_hand_position[0]),
                    int(self.gesture_hand_position[1]),
                )
                if self.diffusion.is_active():
                    frame = self.diffusion.draw_voxels(
                        frame, cube_pos, self.cube_rotation, size=40
                    )
                else:
                    frame = self.draw_image_on_cube(
                        frame, cube_pos, size=40, rotation=self.projection_rotation
                    )
                if self.show_annotations:
                    self.draw_3d_cube(frame, cube_pos, self.cube_rotation, size=40)

            status = f"Image {self.image_index + 1}/{len(self.images)} Active"
            if self.diffusion.is_active():
                mode = (
                    "Exploding"
                    if self.diffusion.mode == "explode"
                    else "Reconstructing"
                )
                status += f" | {mode}"
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
            elif key == ord("h"):
                self.show_annotations = not self.show_annotations
                print(f"Annotations: {'ON' if self.show_annotations else 'OFF'}")

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
