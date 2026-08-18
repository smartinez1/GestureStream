"""
Gesture Stream Showcase - all four effects running simultaneously.

Fire (an L sign lights it at that palm; the other hand's vertical position
is the heat — high hand = bigger fire, low = smaller, no second hand =
fades out), a growable tree (hands spread apart -> together, palms up),
the kaleidoscope (prayer-pose hold toggles) and the blue/purple
color-diffusion trail (three both-hand swings toggles) all render every
frame on top of the same camera feed, sharing one HandLandmarker and one
PersonSegmenter.

The controller is a thin orchestrator: all gesture logic is delegated to
the gesture-state classes imported from the effects/ scripts
(FireGestureState, TreeGestureState, KaleidoscopeGestureState,
WaveGestureState, PersonTrackerState) — nothing is duplicated here.

Usage:
    python3 gesture_stream_showcase.py          # camera
    python3 gesture_stream_showcase.py --demo   # synthetic demo, no camera
"""

import argparse
import math
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "effects"))

import cv2
import mediapipe as mp
import numpy as np

from gesture_stream_fire import FireEffect, FireGestureState
from gesture_stream_tree import VoxelTree, TreeGestureState
from gesture_stream_effects import (
    Kaleidoscope,
    PersonSegmenter,
    ColorDiffusion,
    KaleidoscopeGestureState,
    WaveGestureState,
    PersonTrackerState,
)


class ShowcaseController:
    """Simultaneous effects: one shared hand landmarker feeds all four
    gesture-state objects every frame; one shared person segmenter feeds
    the kaleidoscope anchor and the trail silhouette. Render order:
    kaleidoscope remap -> color trail paint -> tree -> fire -> HUD."""

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
        self.tree = VoxelTree()
        self.kaleido = Kaleidoscope()
        self.diffusion = ColorDiffusion()
        self.segmenter = PersonSegmenter()

        # all gesture logic lives in the state classes (no duplication)
        self.fire_gesture = FireGestureState()
        self.tree_gesture = TreeGestureState()
        self.kaleido_gesture = KaleidoscopeGestureState()
        self.wave_gesture = WaveGestureState()
        self.person_tracker = PersonTrackerState(self.segmenter)

        self.frame_hands = []
        self.frame_size = (0, 0)
        self.show_hud = True
        self._last_render_t = None

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

    # ---- frame pipeline ----

    def process_frame(self, frame, frame_timestamp_ms):
        """Detect hands once, delegate all gesture logic to the imported
        state classes, and return the frame with every effect applied in
        fixed order: kaleidoscope remap -> color trail paint -> tree ->
        fire."""
        h, w = frame.shape[:2]
        self.frame_size = (h, w)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        self.frame_hands = [list(lm) for lm in result.hand_landmarks]
        handedness = [hr[0].category_name for hr in result.handedness]

        now = frame_timestamp_ms / 1000.0
        render_t = time.time()
        dt = (
            min(render_t - self._last_render_t, 0.1)
            if self._last_render_t is not None
            else 1.0 / 30.0
        )
        self._last_render_t = render_t

        trigger_center, heat, burning = self.fire_gesture.update(
            self.frame_hands, (h, w)
        )
        if burning:
            self.fire.ignite()
        else:
            self.fire.extinguish()

        self.tree_gesture.update(self.frame_hands, handedness, (h, w), now, self.tree)
        self.kaleido_gesture.update(self.frame_hands, now)
        self.wave_gesture.update(self.frame_hands, now)

        # a. kaleidoscope remap on the raw feed (fades out smoothly too)
        if self.kaleido_gesture.enabled or self.kaleido.blend > 0.01:
            if self.person_tracker.person_rect is not None:
                frame = self.kaleido.apply(
                    frame,
                    dt=dt,
                    box=self.person_tracker.person_rect,
                    active=self.kaleido_gesture.enabled,
                )
            else:
                frame = self.kaleido.apply(
                    frame, dt=dt, active=self.kaleido_gesture.enabled
                )

        # b. color trail painted on top of the remap
        if self.wave_gesture.wave_enabled or self.diffusion.is_active():
            frame = self.diffusion.apply(
                frame,
                self.segmenter.last_mask,
                now,
                dt=dt,
                enabled=self.wave_gesture.wave_enabled,
            )

        # c. tree (3D element)
        if self.tree.is_active():
            rotation = np.array(
                [
                    0.22 * np.sin(render_t * 0.7),
                    0.30 * np.cos(render_t * 0.5),
                    0.15 * np.sin(render_t * 0.9),
                ]
            )
            frame = self.tree.draw(
                frame,
                (
                    self.tree_gesture.anchor[0],
                    self.tree_gesture.anchor[1] - self.tree_gesture.ANCHOR_UP_OFFSET,
                ),
                rotation,
                dt=dt,
            )

        # d. fire (particle overlay on top of everything)
        if trigger_center is not None:
            intensity = heat * self.fire_gesture.flame
            frame = self.fire.draw(frame, trigger_center, heat=intensity, dt=dt)

        return frame

    def _draw_hud(self, frame, fps):
        """Semi-transparent status panel: active effects, gesture hints,
        FPS."""
        panel = np.zeros_like(frame)
        cv2.rectangle(panel, (10, 10), (430, 205), (40, 40, 40), -1)
        cv2.rectangle(panel, (10, 10), (430, 205), (0, 255, 0), 1)
        frame = cv2.addWeighted(frame, 1.0, panel, 0.6, 0)

        if self.tree.is_active():
            if self.tree.growing:
                tree_state = "GROWING"
            elif self.tree.fading:
                tree_state = "FADING"
            else:
                tree_state = "HELD"
        else:
            tree_state = "IDLE"
        fire_state = "BURNING" if self.fire.is_burning() else "READY"
        fire_intensity = self.fire_gesture.heat * self.fire_gesture.flame
        kaleido_state = "ON" if self.kaleido_gesture.enabled else "OFF"
        trail_state = "ON" if self.wave_gesture.wave_enabled else "OFF"

        lines = [
            f"Fire: {fire_state} | Intensity: {fire_intensity * 100:.0f}%",
            f"Tree: {tree_state} | Palms up: "
            f"{'YES' if self.tree_gesture.palms_up else 'NO'}",
            f"Kaleidoscope: {kaleido_state}",
            f"Color trail: {trail_state}",
            f"FPS: {fps:.1f}",
            "q: quit | h: HUD",
        ]
        y = 36
        for line in lines:
            cv2.putText(
                frame,
                line,
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 255),
                1,
            )
            y += 26

        return frame

    def run(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("Gesture Stream Showcase Active!")
        print("- Make an L sign: fire burns at that palm "
              "(raise/lower your other hand to control the heat)")
        print("- Hold your hands apart, then bring them together (palms up): "
              "a tree grows")
        print("- Hold your hands together as if praying (0.6s): "
              "toggle kaleidoscope")
        print("- Wave with both hands (3 swings): toggle color trail")
        print("- 'h' to toggle HUD")
        print("- 'q' to quit")

        start_time = time.time()
        last_draw = time.time()
        frame_count = 0
        fps = 0.0
        fps_window_start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_timestamp_ms = int((time.time() - start_time) * 1000)
            # segment the raw feed first so the kaleidoscope box and trail
            # mask never come from an already-remapped frame
            self.person_tracker.update(frame)
            frame = self.process_frame(frame, frame_timestamp_ms)

            now = time.time()
            dt = min(now - last_draw, 0.1)
            last_draw = now
            frame_count += 1
            if frame_count % 30 == 0:
                # average wall-time over the last 30 frames
                elapsed = now - fps_window_start
                fps = 30.0 / elapsed if elapsed > 0 else 0.0
                fps_window_start = now

            if self.show_hud:
                frame = self._draw_hud(frame, fps)

            cv2.imshow("Gesture Stream Showcase", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("h"):
                self.show_hud = not self.show_hud
                print(f"HUD: {'ON' if self.show_hud else 'OFF'}")

        cap.release()
        cv2.destroyAllWindows()


def demo(frames=None):
    """Synthetic no-camera demo: a stylized person on the left and two
    palms on the right drive all four effects at once (kaleidoscope and
    color trail cycle on timers, the fire burns at one palm with
    sine-modulated heat, and the tree grows between the palms)."""
    print("Gesture Stream Showcase Demo (No Camera Required)")
    print("Press 'q' to quit\n")

    h, w = 720, 1280
    background = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(w):
        background[:, i] = [int(i / w * 50), int(i / w * 100), 100]

    fire = FireEffect()
    fire.ignite()
    tree = VoxelTree()
    kaleido = Kaleidoscope()
    diffusion = ColorDiffusion()

    start = time.time()
    frame_count = 0
    tree_spawned = False

    while frames is None or frame_count < frames:
        t = time.time() - start
        cyc = t % 16.0
        kaleido_enabled = cyc < 8.0
        trail_enabled = 4.0 <= cyc < 12.0
        heat = float(0.5 + 0.5 * np.sin(t * 0.9))

        # palms on the right trace the tree gesture:
        # spread apart -> together (tree grows) -> spread apart (fades)
        base_x, py = int(w * 0.74), int(h * 0.62)
        if cyc < 3.0:
            px1, px2 = base_x - 300, base_x + 300
            phase = "spread"
        elif cyc < 5.0:
            f = (cyc - 3.0) / 2.0
            px1 = int(base_x - 300 + f * 270)
            px2 = int(base_x + 300 - f * 270)
            phase = "together"
        elif cyc < 8.0:
            px1, px2 = base_x - 30, base_x + 30
            phase = "held"
        elif cyc < 10.0:
            f = (cyc - 8.0) / 2.0
            px1 = int(base_x - 30 - f * 270)
            px2 = int(base_x + 30 + f * 270)
            phase = "release"
        else:
            px1, px2 = base_x - 300, base_x + 300
            phase = "released"
        if phase in ("together", "held"):
            py = py + 30

        if cyc >= 5.0 and cyc < 8.0 and not tree.is_active() and not tree_spawned:
            tree.spawn(height_scale=1.0)
            tree_spawned = True
        elif cyc >= 8.0 and tree.is_active() and not tree.fading:
            tree.start_fade()
        if cyc < 5.0:
            tree_spawned = False

        frame = background.copy()
        mask = np.zeros((h, w), dtype=np.uint8)

        # stylized person on the left (kaleidoscope box + trail silhouette)
        px = int(w * 0.28 + math.sin(t * 0.5) * 80)
        pyp = int(h * 0.42)
        body_w, body_h = 150, 280
        head_r = 45
        cv2.circle(frame, (px, pyp - body_h // 2 + 10), head_r, (120, 160, 220), -1)
        cv2.circle(mask, (px, pyp - body_h // 2 + 10), head_r, 255, -1)
        cv2.rectangle(
            frame,
            (px - body_w // 2, pyp - body_h // 2 + 10 + head_r),
            (px + body_w // 2, pyp + body_h // 2),
            (160, 200, 240),
            -1,
        )
        cv2.rectangle(
            mask,
            (px - body_w // 2, pyp - body_h // 2 + 10 + head_r),
            (px + body_w // 2, pyp + body_h // 2),
            255,
            -1,
        )
        for dx in (-40, 40):
            cv2.line(
                frame,
                (px + dx, pyp + body_h // 2),
                (px + int(dx * 1.5), pyp + body_h // 2 + 60),
                (160, 200, 240),
                18,
            )
            cv2.line(
                mask,
                (px + dx, pyp + body_h // 2),
                (px + int(dx * 1.5), pyp + body_h // 2 + 60),
                255,
                18,
            )
        x0, y0 = px - body_w // 2, pyp - body_h // 2
        x1, y1 = px + body_w // 2, pyp + body_h // 2 + 60

        # render pipeline (same order as the camera path)
        if kaleido_enabled or kaleido.blend > 0.01:
            frame = kaleido.apply(
                frame,
                dt=1.0 / 30.0,
                box=(x0, y0, x1, y1),
                active=kaleido_enabled,
            )
        if trail_enabled or diffusion.is_active():
            frame = diffusion.apply(
                frame, mask.astype(bool), t, dt=1.0 / 30.0, enabled=trail_enabled
            )
        anchor = ((px1 + px2) // 2, py)
        if tree.is_active():
            rotation = np.array(
                [
                    0.22 * np.sin(t * 0.7),
                    0.30 * np.cos(t * 0.5),
                    0.15 * np.sin(t * 0.9),
                ]
            )
            frame = tree.draw(
                frame, (anchor[0], anchor[1] - 40), rotation, dt=1.0 / 30.0
            )
        frame = fire.draw(frame, (px2, py), heat=heat, dt=1.0 / 30.0)

        # palms following the gesture path
        for x in (px1, px2):
            cv2.circle(frame, (x, py), 38, (90, 90, 160), 3)
            cv2.circle(frame, (x, py), 30, (40, 40, 120), -1)

        k_state = "ON" if kaleido_enabled else "OFF"
        t_state = "ON" if trail_enabled else "OFF"
        cv2.putText(
            frame,
            f"Kaleidoscope: {k_state} | Trail: {t_state} | "
            f"Fire heat: {heat * 100:.0f}% | Phase: {phase.upper()}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 255),
            2,
        )

        cv2.imshow("Gesture Stream Showcase Demo", frame)
        key = cv2.waitKey(33) & 0xFF
        if key == ord("q"):
            break
        frame_count += 1

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Gesture-controlled showcase: fire, tree, kaleidoscope, "
        "color trail"
    )
    parser.add_argument("--demo", action="store_true", help="synthetic demo, no camera")
    args = parser.parse_args()

    if args.demo:
        demo()
    else:
        controller = ShowcaseController()
        controller.run()


if __name__ == "__main__":
    main()