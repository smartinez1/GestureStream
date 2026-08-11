"""
Gesture Stream Fire - standalone gesture-controlled fire effect.

Trigger: an open hand pointing up ("open hand pointing up flat") starts a
flame at the palm center of that hand. The flame is continuously modulated
by the OTHER hand's openness: open hand = roaring fire, closed = embers.

Self-contained: does not import any other gesture_stream module. Uses the
MediaPipe Task API directly (VIDEO mode, 2 hands) and renders a numpy
particle system with a blurred, additively-blended glow buffer.

Usage:
    python3 gesture_stream_fire.py            # camera
    python3 gesture_stream_fire.py --demo     # synthetic demo, no camera
"""

import argparse
import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np


class FireEffect:
    """Vectorized particle-system fire with an additive glow pass."""

    MAX_PARTICLES = 800
    SPAWN_RATE = 300.0          # particles per second at heat=1.0
    GLOW_SCALE = 4              # glow buffer = frame / GLOW_SCALE
    HALO_BLUR = 12              # gaussian sigma for the additive halo
    BODY_BLUR = 4               # gaussian sigma for the opaque flame body
    CORE_OPACITY = 0.9          # how solidly the body replaces the background
    BODY_RADIUS_MULT = 1.7      # body pass drawn larger than the halo pass
    BUOYANCY = 70.0             # px/s^2 deceleration of the rising draft
    DRAG = 1.4                  # 1/s lateral drag on vx
    TURBULENCE = 90.0           # px/s^2 random lateral acceleration (heat-scaled)
    SPEED_MIN = 120.0           # px/s upward, at heat=1.0
    SPEED_MAX = 300.0
    SPREAD = 55.0               # px/s lateral jitter at spawn
    LIFETIME_MIN = 0.8          # s
    LIFETIME_MAX = 1.6
    RADIUS_MIN = 5.0            # px, heat-scaled
    RADIUS_MAX = 14.0
    EMITTER_RADIUS = 18.0       # px jitter around the anchor point
    BRIGHTNESS = 1.5            # color multiplier for the additive glow

    # Flame color ramp over particle age f in [0,1] (BGR).
    # White core -> yellow -> orange -> deep red -> dark ash.
    COLOR_STOPS = np.array(
        [
            [0.00, 255, 255, 255],
            [0.10, 250, 255, 255],
            [0.28, 150, 235, 255],
            [0.55, 55, 130, 235],
            [0.80, 25, 60, 160],
            [1.00, 10, 20, 70],
        ],
        dtype=np.float32,
    )

    def __init__(self):
        self.rng = np.random.default_rng()
        self.positions = np.zeros((0, 2), dtype=np.float32)
        self.velocities = np.zeros((0, 2), dtype=np.float32)
        self.lifetimes = np.zeros(0, dtype=np.float32)
        self.max_lifetimes = np.zeros(0, dtype=np.float32)
        self.radii = np.zeros(0, dtype=np.float32)
        self.burning = False

    def ignite(self):
        self.burning = True

    def extinguish(self):
        """Stop spawning; existing particles fade out naturally."""
        self.burning = False

    def is_burning(self):
        return self.burning

    def has_particles(self):
        return len(self.positions) > 0

    def _spawn(self, center, heat):
        n = int(self.SPAWN_RATE * heat * (1.0 / 30.0))
        n = min(n, self.MAX_PARTICLES - len(self.positions))
        if n <= 0:
            return
        speed = self.rng.uniform(self.SPEED_MIN, self.SPEED_MAX, n) * (0.4 + 0.7 * heat)
        pos = np.empty((n, 2), dtype=np.float32)
        pos[:, 0] = center[0] + self.rng.uniform(-1, 1, n) * self.EMITTER_RADIUS
        pos[:, 1] = center[1] + self.rng.uniform(-1, 1, n) * self.EMITTER_RADIUS
        vel = np.empty((n, 2), dtype=np.float32)
        vel[:, 0] = self.rng.uniform(-self.SPREAD, self.SPREAD, n)
        vel[:, 1] = -speed
        life = self.rng.uniform(self.LIFETIME_MIN, self.LIFETIME_MAX, n)
        radius = self.rng.uniform(self.RADIUS_MIN, self.RADIUS_MAX, n) * (0.6 + 0.8 * heat)

        self.positions = np.concatenate([self.positions, pos], axis=0)
        self.velocities = np.concatenate([self.velocities, vel], axis=0)
        self.lifetimes = np.concatenate([self.lifetimes, life])
        self.max_lifetimes = np.concatenate([self.max_lifetimes, life])
        self.radii = np.concatenate([self.radii, radius])

    def _update(self, dt):
        if len(self.positions) == 0:
            return
        self.lifetimes -= dt
        self.velocities[:, 1] += self.BUOYANCY * dt
        self.velocities[:, 0] *= np.exp(-self.DRAG * dt)
        self.velocities[:, 0] += self.rng.normal(0.0, self.TURBULENCE * dt, len(self.positions))

        alive = self.lifetimes > 0
        if not np.all(alive):
            self.positions = self.positions[alive]
            self.velocities = self.velocities[alive]
            self.lifetimes = self.lifetimes[alive]
            self.max_lifetimes = self.max_lifetimes[alive]
            self.radii = self.radii[alive]

        self.positions += self.velocities * dt

    def _color_ramp(self, age_fraction):
        """BGR colors (float32, 0-255) for per-particle age fractions."""
        out = np.empty((len(age_fraction), 3), dtype=np.float32)
        for c in range(3):
            out[:, c] = np.interp(
                age_fraction, self.COLOR_STOPS[:, 0], self.COLOR_STOPS[:, c + 1]
            )
        return out * self.BRIGHTNESS

    def draw(self, frame, center, heat=1.0, dt=1.0 / 30.0):
        """Update and render the fire. heat in [0,1] modulates the whole
        effect (spawn rate, speed, size). Two passes: an opaque body
        (solid flame color replaces the background) plus an additive halo
        for the glare."""
        if self.burning:
            self._spawn(center, heat)
        self._update(dt)
        n = len(self.positions)
        if n == 0:
            return frame

        h, w = frame.shape[:2]
        s = self.GLOW_SCALE
        body_img = np.zeros((h // s, w // s, 3), dtype=np.uint8)
        halo_img = np.zeros((h // s, w // s, 3), dtype=np.uint8)

        age = 1.0 - self.lifetimes / np.maximum(self.max_lifetimes, 1e-6)
        colors = self._color_ramp(age)
        scale = 1.0 / s
        for i in range(n):
            px = int(self.positions[i, 0] * scale)
            py = int(self.positions[i, 1] * scale)
            body_r = max(1, int(self.radii[i] * self.BODY_RADIUS_MULT * scale))
            halo_r = max(1, int(self.radii[i] * scale))
            color = tuple(int(c) for c in colors[i])
            cv2.circle(body_img, (px, py), body_r, color, -1)
            cv2.circle(halo_img, (px, py), halo_r, color, -1)

        body = cv2.resize(
            cv2.GaussianBlur(body_img, (0, 0), self.BODY_BLUR),
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )
        halo = cv2.resize(
            cv2.GaussianBlur(halo_img, (0, 0), self.HALO_BLUR),
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )

        # Opaque composite: the flame body's own brightness is its alpha,
        # so the flame color replaces the background instead of adding to it.
        mask = (
            (body.max(axis=2).astype(np.float32) / 255.0) ** 1.2
            * self.CORE_OPACITY
        )
        mask3 = np.clip(mask, 0.0, 1.0)[..., None]
        composite = (
            frame.astype(np.float32) * (1.0 - mask3)
            + body.astype(np.float32) * mask3
        )
        frame = np.clip(composite, 0, 255).astype(np.uint8)
        return cv2.add(frame, halo)


class FireGestureController:
    """Standalone camera controller: open-hand-pointing-up triggers the
    fire, other hand's openness continuously modulates its intensity."""

    OPENNESS_MIN = 0.10        # openness (tip/PIP ratio) -> heat 0.0
    OPENNESS_MAX = 0.55        # openness -> heat 1.0
    HEAT_SMOOTHING = 0.12      # per-frame ease toward the target heat
    VERTICALITY_THRESHOLD = 1.2
    # wrist->middle-tip vector normalized by palm size; negative y points
    # up (screen y grows downward). 1.2 = fingers clearly above the wrist.
    FINGER_EXTEND_OFFSET = 0.03

    def __init__(self):
        self.BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        model_path = self._get_model_path()
        options = self.HandLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=model_path),
            running_mode=self.VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hand_landmarker = self.HandLandmarker.create_from_options(options)

        self.fire = FireEffect()
        self.trigger_center = None
        self.heat = 0.05
        self.show_annotations = True
        self.frame_hands = []

    def _get_model_path(self):
        """Get or download the hand landmarker model (same location the
        other scripts use: ~/.mediapipe/models)."""
        model_dir = os.path.expanduser("~/.mediapipe/models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "hand_landmarker.task")
        if os.path.exists(model_path):
            print(f"Using cached model: {model_path}")
            return model_path
        print("Downloading hand landmarker model...")
        url = (
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        )
        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"Model downloaded to: {model_path}")
            return model_path
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise

    def _hand_openness(self, landmarks):
        """Mean tip-to-PIP distance relative to palm size (scale invariant)."""
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        palm = max(
            np.linalg.norm([wrist.x - middle_mcp.x, wrist.y - middle_mcp.y]),
            1e-6,
        )
        distances = []
        for tip_idx, pip_idx in ((4, 3), (8, 6), (12, 10), (16, 14), (20, 18)):
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            distances.append(np.linalg.norm([tip.x - pip.x, tip.y - pip.y]))
        return float(np.mean(distances) / palm)

    def _normalize_openness(self, openness):
        return float(
            np.clip(
                (openness - self.OPENNESS_MIN)
                / (self.OPENNESS_MAX - self.OPENNESS_MIN),
                0.0,
                1.0,
            )
        )

    def _is_open(self, landmarks):
        """>= 4 of 5 fingers extended (same logic as the core scripts)."""
        extended = 0
        for tip_idx, pip_idx in ((4, 3), (8, 6), (12, 10), (16, 14), (20, 18)):
            if landmarks[tip_idx].y < landmarks[pip_idx].y - self.FINGER_EXTEND_OFFSET:
                extended += 1
        return extended >= 4

    def _points_up(self, landmarks):
        """Open hand held flat, fingers pointing up (screen y grows down)."""
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        middle_tip = landmarks[12]
        palm = max(
            np.linalg.norm([middle_mcp.x - wrist.x, middle_mcp.y - wrist.y]),
            1e-6,
        )
        d_y = (middle_tip.y - wrist.y) / palm
        return d_y < -self.VERTICALITY_THRESHOLD

    def process_frame(self, frame, frame_timestamp_ms):
        """Classify hands: the first open-pointing-up hand is the flame
        anchor; the other hand's openness modulates intensity."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        h, w = frame.shape[:2]
        self.frame_hands = []
        self.trigger_center = None
        trigger = None
        modulator = None

        if result.hand_landmarks:
            hands = [list(landmarks) for landmarks in result.hand_landmarks]
            self.frame_hands = hands
            for landmarks in hands:
                if trigger is None and self._is_open(landmarks) and self._points_up(landmarks):
                    trigger = landmarks
                elif modulator is None:
                    modulator = landmarks
            if trigger is not None:
                palm = np.mean([[lm.x for lm in trigger], [lm.y for lm in trigger]], axis=1)
                self.trigger_center = (int(palm[0] * w), int(palm[1] * h))
                source = modulator if modulator is not None else trigger
                target = self._normalize_openness(self._hand_openness(source))
            else:
                target = self.heat
        else:
            target = self.heat

        self.heat += (target - self.heat) * self.HEAT_SMOOTHING
        if trigger is not None:
            self.fire.ignite()
        else:
            self.fire.extinguish()

    def run(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("Gesture Stream Fire Active!")
        print("- Hold an open hand pointing up: fire burns at the palm")
        print("- Openness of your OTHER hand controls the intensity")
        print("- 'h' to toggle hand annotations")
        print("- 'q' to quit")

        start_time = time.time()
        last_fire_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_timestamp_ms = int((time.time() - start_time) * 1000)
            self.process_frame(frame, frame_timestamp_ms)

            now = time.time()
            dt = min(now - last_fire_time, 0.1)
            last_fire_time = now

            if self.trigger_center is not None:
                frame = self.fire.draw(
                    frame, self.trigger_center, heat=self.heat, dt=dt
                )

            if self.show_annotations:
                for landmarks in self.frame_hands:
                    for lm in landmarks:
                        cv2.circle(
                            frame,
                            (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])),
                            3,
                            (0, 255, 0),
                            -1,
                        )

            state = "BURNING" if self.fire.is_burning() else "FADING"
            cv2.putText(
                frame,
                f"Fire: {state} | Heat: {self.heat * 100:.0f}%",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 220, 255),
                2,
            )

            cv2.imshow("Gesture Stream Fire", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("h"):
                self.show_annotations = not self.show_annotations
                print(f"Annotations: {'ON' if self.show_annotations else 'OFF'}")

        cap.release()
        cv2.destroyAllWindows()


def demo(frames=None):
    """Synthetic no-camera demo: sine-modulated heat."""
    print("Gesture Stream Fire Demo (No Camera Required)")
    print("Press 'q' to quit\n")

    h, w = 720, 1280
    background = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(w):
        background[:, i] = [int(i / w * 50), int(i / w * 100), 100]

    fire = FireEffect()
    fire.ignite()
    center = (w // 2, h // 2 + 100)
    start = time.time()
    frame_count = 0

    while frames is None or frame_count < frames:
        t = time.time() - start
        heat = float(0.5 + 0.5 * np.sin(t * 0.9))
        frame = background.copy()
        frame = fire.draw(frame, center, heat=heat, dt=1.0 / 30.0)

        cv2.putText(
            frame,
            f"Heat: {heat * 100:.0f}%",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 255),
            2,
        )
        cv2.imshow("Gesture Stream Fire Demo", frame)

        key = cv2.waitKey(33) & 0xFF
        if key == ord("q"):
            break
        frame_count += 1

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Gesture-controlled fire effect")
    parser.add_argument("--demo", action="store_true", help="synthetic demo, no camera")
    args = parser.parse_args()

    if args.demo:
        demo()
    else:
        controller = FireGestureController()
        controller.run()


if __name__ == "__main__":
    main()