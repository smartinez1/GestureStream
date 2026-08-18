"""
Gesture Stream Diffusion - bursts a projected image into expanding 3D voxels
colored by the image palette (no image texture is drawn, only the voxels).

The core controller (gesture_stream.py) triggers this on the closed -> open
transition of the non-pointing hand. This module is also runnable standalone
as a synthetic demo (no camera required).

Usage:
    python3 gesture_stream_diffusion.py
"""

import cv2
import numpy as np
import time
from PIL import Image


class VoxelDiffusion:
    """Diffusion burst: image palette -> expanding, fading 3D voxels."""

    DURATION = 5.0
    EXPLODE_CAP = 0.6

    def __init__(self, num_voxels=64):
        self.num_voxels = num_voxels
        self.image = None
        self.palette = []
        self.voxels = []
        self.mode = "explode"
        self.active = False
        self.start_time = 0.0
        self.duration = self.DURATION
        self.start_state = 0.0
        self.rng = np.random.default_rng()

    def trigger(self, image_bgr, duration=DURATION):
        """Start the explosion burst from a BGR image."""
        self.image = image_bgr
        self.palette = self._extract_palette(image_bgr)
        self.voxels = self._generate_voxels()
        self.mode = "explode"
        self.active = True
        self.start_time = time.time()
        self.duration = float(duration)
        self.start_state = 0.0

    def reconstruct(self, image_bgr, duration=DURATION):
        """Reassemble: converge the voxels while morphing their colors
        toward the new image's palette. Starts from the explosion state
        at the moment the hand closes, so there is no jump to a default
        size."""
        if self.mode == "explode" and self.active:
            self.start_state = min(
                self.elapsed() / self.duration, self.EXPLODE_CAP
            )
        else:
            self.start_state = 0.0
        self.image = image_bgr
        self.palette = self._extract_palette(image_bgr)
        if not self.voxels:
            self.voxels = self._generate_voxels()
        for voxel in self.voxels:
            voxel["color_from"] = voxel["color"]
            r, g, b = self.palette[int(self.rng.integers(0, len(self.palette)))]
            voxel["color_to"] = (b, g, r)
        self.mode = "reconstruct"
        self.active = True
        self.start_time = time.time()
        self.duration = float(duration)

    def is_active(self):
        """True while the animation is still playing. An explosion stays
        active (held at its max expansion) until reconstruct is triggered."""
        if not self.active:
            return False
        if self.mode == "explode":
            return True
        if time.time() - self.start_time >= self.duration:
            self.active = False
        return self.active

    def elapsed(self):
        """Seconds since trigger, clamped to the animation duration."""
        if not self.active:
            return 0.0
        return min(time.time() - self.start_time, self.duration)

    def progress(self):
        """Animation progress in [0.0, 1.0]."""
        return self.elapsed() / self.duration

    def _extract_palette(self, image_bgr, colors=10):
        """Dominant colors via PIL median-cut quantization (RGB triplets).

        Near-white and near-black pixels are excluded and the rest is
        sampled weighted by saturation, so backgrounds and flat grays
        don't swallow the image's actual colors.
        """
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pixels = rgb.reshape(-1, 3)
        min_ch = pixels.min(axis=1)
        max_ch = pixels.max(axis=1)
        colorful = pixels[(min_ch < 220) & (max_ch > 30)]
        if len(colorful) < 200:
            colorful = pixels

        saturation = colorful.max(axis=1) - colorful.min(axis=1)
        if saturation.max() > 0:
            weights = saturation.astype(np.float64) ** 2 + 1.0
            weights = weights / weights.sum()
            sample = colorful[
                self.rng.choice(
                    len(colorful),
                    size=min(20000, len(colorful)),
                    replace=True,
                    p=weights,
                )
            ]
        else:
            sample = colorful

        image = Image.fromarray(sample.reshape(1, -1, 3))
        quantized = image.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
        palette = quantized.getpalette()[: colors * 3]
        return [
            (palette[i], palette[i + 1], palette[i + 2])
            for i in range(0, len(palette), 3)
        ]

    def _generate_voxels(self):
        """Voxels on a sphere shell with random speed, size and palette color."""
        voxels = []
        for _ in range(self.num_voxels):
            direction = self.rng.standard_normal(3)
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            speed = 20.0 + self.rng.uniform(0.0, 60.0)
            size = int(self.rng.integers(4, 10))
            r, g, b = self.palette[int(self.rng.integers(0, len(self.palette)))]
            voxels.append(
                {
                    "direction": direction,
                    "speed": speed,
                    "size": size,
                    "color": (b, g, r),  # BGR for OpenCV frames
                }
            )
        return voxels

    def _rotation_matrix(self, rotation):
        """Rotation matrix for the cube orientation applied to voxel offsets."""
        rx, ry, rz = rotation
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return rot_z @ rot_y @ rot_x

    def draw(self, frame, center, rotation, size=50):
        """Draw the voxel animation (explode or reconstruct)."""
        if not self.is_active():
            return frame
        return self.draw_voxels(frame, center, rotation, size)

    def draw_voxels(self, frame, center, rotation, size=50):
        """Draw the voxels: expanding/fading (explode) or converging with
        morphing colors (reconstruct)."""
        if not self.is_active():
            return frame
        h, w = frame.shape[:2]
        progress = self.progress()
        life = 1.0 - progress
        rot_matrix = self._rotation_matrix(rotation)

        for voxel in self.voxels:
            direction = rot_matrix @ voxel["direction"]

            if self.mode == "reconstruct":
                start = self.start_state
                start_radius = size * 0.6 + start * voxel["speed"]
                radius = start_radius * (1.0 - progress) + size * 0.25 * progress
                start_half = voxel["size"] * (1.0 - 0.4 * start)
                half = max(
                    1, int(start_half * (1.0 - progress) + voxel["size"] * 1.4 * progress)
                )
                t = progress * progress * (3.0 - 2.0 * progress)
                color = (
                    np.array(voxel["color_from"], dtype=np.float32) * (1.0 - t)
                    + np.array(voxel["color_to"], dtype=np.float32) * t
                )
                start_alpha = 0.9 * (1.0 - start)
                alpha = start_alpha * (1.0 - progress) + 0.9 * progress
            else:
                p = min(progress, self.EXPLODE_CAP)
                radius = size * 0.6 + p * voxel["speed"]
                half = max(1, int(voxel["size"] * (1.0 - 0.4 * p)))
                color = np.array(voxel["color"], dtype=np.float32)
                alpha = 0.9 * (1.0 - p)

            px = int(center[0] + direction[0] * radius)
            py = int(center[1] + direction[1] * radius)

            x1, y1 = px - half, py - half
            x2, y2 = px + half, py + half
            if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
                continue
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            roi = frame[y1:y2, x1:x2].astype(np.float32)
            frame[y1:y2, x1:x2] = (roi * (1.0 - alpha) + color * alpha).astype(
                np.uint8
            )
        return frame


def main():
    """Synthetic demo: burst a generated gradient image into voxels."""
    print("Gesture Stream Diffusion Demo (No Camera Required)")
    print("Press 'q' to quit\n")

    h, w = 720, 1280
    background = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(w):
        background[:, i] = [int(i / w * 50), int(i / w * 100), 100]

    image = np.zeros((200, 200, 3), dtype=np.uint8)
    for i in range(200):
        image[:, i] = [int(i / 200 * 255), 100, 150]

    diffusion = VoxelDiffusion()
    diffusion.trigger(image)
    center = (w // 2, h // 2)
    start = time.time()

    next_image = np.zeros((200, 200, 3), dtype=np.uint8)
    for i in range(200):
        next_image[:, i] = [150, int(i / 200 * 255), 100]

    reconstruct_at = start + 3.0
    end_at = start + 3.0 + VoxelDiffusion.DURATION + 1.0

    while time.time() < end_at:
        if time.time() >= reconstruct_at and diffusion.mode == "explode":
            diffusion.reconstruct(next_image)
        frame = background.copy()
        rotation = np.array(
            [
                np.sin(time.time() - start) * 0.4,
                np.cos(time.time() - start) * 0.5,
                (time.time() - start) * 0.8,
            ]
        )
        frame = diffusion.draw(frame, center, rotation, size=40)
        cv2.imshow("Gesture Stream Diffusion Demo", frame)
        if cv2.waitKey(16) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
