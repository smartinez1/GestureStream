"""
Gesture Stream Effects - gesture-activated visual effects.

Two effects, each self-contained and toggled independently:

1. Kaleidoscope: hold both hands together as if praying (fingers pointing
   up, palms together, thumbs touching) for HOLD_TIME to toggle the filter.
   A MediaPipe selfie-segmentation model runs every few frames; the
   person's bounding box is cropped, mirrored into a 2x2 block, and tiled
   across the frame at 1:1 scale, so you see mirrored copies of your whole
   body fanning out around you while you stay visible in the middle (no
   zooming, no noise). The pattern slowly rotates around the person.

2. Color diffusion trail: wave with both hands (swing them side to side a
   few times) to toggle the effect. Blue-to-purple color is painted in a
   rim around the segmented silhouette and additively accumulates into a
   trail buffer that blurs outward and decays over time, so the silhouette
   leaves a glowing, fading trace of everywhere it has been. Both effects
   stack when enabled together.

The pure gesture logic (prayer pose, wave detector, person-box tracking)
lives in three state classes — KaleidoscopeGestureState, WaveGestureState,
PersonTrackerState — with no MediaPipe or camera, so other scripts (like
the showcase) can import and delegate instead of copying.

Self-contained: does not import any other gesture_stream module. Uses the
MediaPipe Task API directly (VIDEO mode, 2 hands + selfie segmenter).

Usage:
    python3 gesture_stream_effects.py          # camera
    python3 gesture_stream_effects.py --demo   # synthetic demo, no camera
"""

import argparse
import math
import os
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
import numpy as np


class Kaleidoscope:
    """Kaleidoscope built on a cached cv2.remap.

    Two modes:

    * Person mode (``box`` given): the person's bounding box is cropped,
      mirrored into a 2x2 block (left/right + top/bottom reflections), and
      tiled across the frame at 1:1 scale, slowly rotating around the
      person. The person stays sharp in the middle; mirrored copies of
      their whole body fade in around them. No zooming.

    * Frame mode (no ``box``): a radial angle-fold of the whole frame
      (used as the no-person fallback).
    """

    SEGMENTS = 8
    ROTATION_SPEED = 0.5        # rad/s of pattern spin
    FADE_SPEED = 4.0            # blend rate toward the target (95% in ~0.75 s)
    RAMP_START = 0.55           # radius fraction of max where the effect starts
    RAMP_WIDTH = 0.35           # how far past RAMP_START it reaches full strength
    RADIAL_MIRROR = 0.0         # 1.0 = full center<->rim mirror, 0.0 = none
    MASK_FULL = 1.8             # person-mode mask reaches full effect at this x radius
    MARGIN = 1.1                # expand the person box by 5% per side before tiling

    def __init__(self, segments=SEGMENTS, rotation_speed=ROTATION_SPEED,
                 ramp_start=RAMP_START, ramp_width=RAMP_WIDTH,
                 radial_mirror=RADIAL_MIRROR):
        self.segments = segments
        self.rotation_speed = rotation_speed
        self.ramp_start = ramp_start
        self.ramp_width = ramp_width
        self.radial_mirror = radial_mirror
        self.phase = 0.0
        self.blend = 0.0        # 0..1 how far the effect is mixed in (fades)
        self._r = None
        self._theta = None
        self._x = None
        self._y = None
        self._max_r = None
        self._mask = None
        self._grid_key = None

    def _build_grid(self, h, w, cx, cy):
        """Cache the per-pixel coordinate grids for a frame size + center.
        Rebuilt only when size or (rounded) center changes."""
        ys, xs = np.indices((h, w), dtype=np.float32)
        dx = xs - cx
        dy = ys - cy
        self._r = np.sqrt(dx * dx + dy * dy).astype(np.float32)
        self._theta = np.arctan2(dy, dx).astype(np.float32)
        self._x = xs
        self._y = ys
        self._max_r = np.sqrt(
            max(cx, w - cx) ** 2 + max(cy, h - cy) ** 2
        )

    def apply(self, frame, dt=1.0 / 30.0, center=None, box=None, active=None):
        """Return the frame with the kaleidoscope filter applied.

        ``box`` = (x0, y0, x1, y1) of the person. The crop is mirrored and
        tiled across the frame at 1:1 scale (mirrored copies of the whole
        person, no zoom); the person stays visible in the middle and the
        pattern slowly rotates around them. Without ``box`` the frame-wide
        radial fold is used (no-person fallback).

        ``active`` eases the effect in and out instead of snapping: when
        True the blend ramps toward 1.0, when False toward 0.0 (pass it
        every frame, including a few after the toggle turns off, so the
        kaleidoscope fades away smoothly)."""
        if active is not None:
            target = 1.0 if active else 0.0
            self.blend += (target - self.blend) * min(1.0, dt * self.FADE_SPEED)
        h, w = frame.shape[:2]
        if box is not None:
            bx0, by0, bx1, by1 = [float(v) for v in box]
            bw, bh = bx1 - bx0, by1 - by0
            if bw < 4 or bh < 4:
                box = None
            else:
                cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
                hw_, hh_ = bw / 2.0 * self.MARGIN, bh / 2.0 * self.MARGIN
                x0 = max(0, int(cx - hw_))
                x1 = min(w, int(cx + hw_))
                y0 = max(0, int(cy - hh_))
                y1 = min(h, int(cy + hh_))
                if x1 - x0 < 4 or y1 - y0 < 4:
                    box = None
        if box is None:
            if center is None:
                cx, cy = w / 2.0, h / 2.0
            else:
                cx, cy = float(center[0]), float(center[1])
            key = (h, w, int(round(cx)), int(round(cy)), -1)
        else:
            tw, th = x1 - x0, y1 - y0
            key = (h, w, int(round(cx)), int(round(cy)), tw, th)
        if key != self._grid_key:
            self._build_grid(h, w, cx, cy)
            if box is None:
                t = np.clip(
                    (self._r / self._max_r - self.ramp_start) / self.ramp_width,
                    0.0,
                    1.0,
                )
            else:
                keep = max(tw, th) / 2.0
                full = keep * self.MASK_FULL
                t = np.zeros_like(self._r)
                if full > keep:
                    t = np.clip((self._r - keep) / (full - keep), 0.0, 1.0)
            t = t * t * (3.0 - 2.0 * t)          # smoothstep
            self._mask = t[..., None].astype(np.float32)
            self._grid_key = key
        self.phase += dt * self.rotation_speed

        if box is None:
            sector = 2.0 * np.pi / self.segments
            half = sector / 2.0
            folded = np.mod(self._theta + self.phase, sector)
            folded = np.where(folded > half, sector - folded, folded)
            m = self.radial_mirror
            r_src = self._r * (1.0 - m) + (self._max_r - self._r) * m
            map_x = (cx + r_src * np.cos(folded)).astype(np.float32)
            map_y = (cy + r_src * np.sin(folded)).astype(np.float32)
        else:
            # mirror-tile sampling: rotate the output coords around the
            # person, then reflect each one into the [x0,x0+tw]x[y0,y0+th]
            # crop via the 2x2 mirrored block (1:1 scale, no zoom)
            phase = self.phase
            c, s = np.cos(phase), np.sin(phase)
            dx = self._x - cx
            dy = self._y - cy
            rx = cx + dx * c - dy * s
            ry = cy + dx * s + dy * c
            wb, hb = 2.0 * tw, 2.0 * th
            mx = np.mod(rx - x0, wb)
            my = np.mod(ry - y0, hb)
            sx = np.where(mx <= tw, mx, wb - mx)
            sy = np.where(my <= th, my, hb - my)
            map_x = (x0 + sx).astype(np.float32)
            map_y = (y0 + sy).astype(np.float32)

        kaleido = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

        alpha = self._mask * self.blend
        blended = (
            frame.astype(np.float32) * (1.0 - alpha)
            + kaleido.astype(np.float32) * alpha
        )
        return np.clip(blended, 0, 255).astype(np.uint8)


class PersonSegmenter:
    """MediaPipe Selfie Segmentation wrapper: segments the dominant
    foreground person and returns their bounding box. The model downloads
    to ~/.mediapipe/models like the hand landmarker. The person mask is the
    model's confidence output thresholded (raised to drop low-confidence
    background fringing), morphologically closed to fill holes, and reduced
    to its largest connected component so stray false positives can't skew
    the box."""

    MIN_AREA = 0.002           # fraction of frame; smaller boxes are noise
    CONFIDENCE_THRESHOLD = 0.7  # only pixels this confident count as person
    CLOSE_KERNEL = 15           # morphological closing kernel (fills holes)
    MARGIN = 1.1                # pad the radius so head/feet aren't clipped

    def __init__(self):
        model_path = self._get_model_path()
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=True,
            output_confidence_masks=True,
        )
        self.segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options)
        self.last_mask = None   # frame-sized bool mask of the last bbox() call

    def _get_model_path(self):
        """Get or download the selfie segmenter model (same location the
        other scripts use: ~/.mediapipe/models)."""
        model_dir = os.path.expanduser("~/.mediapipe/models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "selfie_segmenter.tflite")
        if os.path.exists(model_path):
            print(f"Using cached model: {model_path}")
            return model_path
        print("Downloading selfie segmenter model...")
        url = (
            "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
            "selfie_segmenter/float16/latest/selfie_segmenter.tflite"
        )
        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"Model downloaded to: {model_path}")
            return model_path
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise

    def bbox(self, frame):
        """Return (x0, y0, x1, y1) of the person, or None if none found.

        The person mask is the model's confidence output for the person
        class, thresholded and morphologically closed. If no confidence
        masks are available it falls back to the category-mask class that
        touches the fewest frame edges (the background spans the borders).
        Also stores `last_mask`, a frame-sized boolean mask of the person,
        for debug overlays."""
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.segmenter.segment(mp_image)

        confs = result.confidence_masks
        if confs:
            conf = np.squeeze(confs[0].numpy_view())
            person = conf >= self.CONFIDENCE_THRESHOLD
        else:
            # fallback: category mask (class that touches fewest edges)
            mask = np.squeeze(result.category_mask.numpy_view())
            mh, mw = mask.shape
            best_key = None
            best = None
            for v in np.unique(mask):
                ys, xs = np.where(mask == v)
                if ys.size == 0:
                    continue
                edges = (
                    int(ys.min() == 0)
                    + int(ys.max() == mh - 1)
                    + int(xs.min() == 0)
                    + int(xs.max() == mw - 1)
                )
                key = (edges, -ys.size)
                if best_key is None or key < best_key:
                    best_key = key
                    best = (v, xs.min(), ys.min(), xs.max(), ys.max())
            if best is None or (best_key[0] >= 4 and np.unique(mask).size == 1):
                self.last_mask = np.zeros((h, w), dtype=bool)
                return None
            person = mask == best[0]

        # morphological closing fills small holes/gaps in the silhouette
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.CLOSE_KERNEL, self.CLOSE_KERNEL)
        )
        person = cv2.morphologyEx(
            person.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ) > 0
        # keep only the largest connected component so stray false-positive
        # blobs (dark screens, glares) can't widen the box
        num, labels = cv2.connectedComponents(person.astype(np.uint8))
        if num > 1:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0                      # background
            person = labels == int(np.argmax(sizes))
        mh, mw = person.shape

        self.last_mask = cv2.resize(
            person.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        ys, xs = np.where(person)
        if ys.size == 0:
            return None
        sx = w / mw if mw != w else 1.0
        sy = h / mh if mh != h else 1.0
        x0, y0 = int(xs.min() * sx), int(ys.min() * sy)
        x1, y1 = int((xs.max() + 1) * sx), int((ys.max() + 1) * sy)
        if (x1 - x0) * (y1 - y0) < self.MIN_AREA * w * h:
            return None
        return (x0, y0, x1, y1)


class ColorDiffusion:
    """Color diffuses outward from the person's silhouette and decays over
    time, leaving a fading trace of everywhere they have been.

    Each active frame a thick rim around the silhouette (a good distance
    in and out from the edge) is additively accumulated into a float32
    trail buffer. The paint strength is strongest right at the silhouette
    edge and falls off quadratically with distance, so the glow fades
    quickly to transparent far from the outline. The whole buffer is
    box-blurred every frame so the color diffuses outward, and multiplied
    by a decay factor so older paint fades away. Trail pixels that lie
    under the person's current segmentation mask get an extra, much
    stronger decay, so paint can never accumulate on the body itself and
    occlude them — the aura lives around the silhouette, not on it. The
    hue of each rim pixel comes from the angle around the silhouette
    centroid plus a time ramp, wrapped into the blue-to-purple band
    (HUE_BASE..HUE_BASE+HUE_RANGE degrees), so the trace reads as a
    flowing blue/indigo/violet aura. When the effect is disabled painting
    stops but the existing trail keeps decaying, so it melts away.
    """

    DECAY = 0.90            # per-frame trail decay (at 30 fps ~0.5 s trace)
    MASK_DECAY = 0.55       # extra decay on trail pixels under the person
    OUTER_PAD = 70          # px the glow extends outside the silhouette
    INNER_PAD = 16          # px the glow extends inside the silhouette
    BLUR = 5                # box-blur kernel that spreads the color
    HUE_SPEED = 55.0        # deg/s the hue cycles around the silhouette
    INTENSITY = 0.4         # additive intensity of each paint stroke
    HUE_BASE = 180.0        # degrees: start of hue band (HSV 0-360; blue)
    HUE_RANGE = 120.0       # degrees: width of band (180-300° = blue -> purple)

    def __init__(self):
        self.trail = None   # float32 (h, w, 3) accumulation buffer

    def _ensure(self, h, w):
        if self.trail is None or self.trail.shape[:2] != (h, w):
            self.trail = np.zeros((h, w, 3), dtype=np.float32)

    def clear(self):
        if self.trail is not None:
            self.trail[:] = 0.0

    def is_active(self):
        """True while the trail still holds visible paint (used to keep
        decaying after the effect is toggled off). Values below ~1 are
        sub-threshold once added to an 8-bit frame."""
        return self.trail is not None and bool((self.trail > 1.0).any())

    def _paint(self, mask, t):
        """Add a thick blue/purple rim around the silhouette to the trail.

        A distance transform gives each rim pixel its distance from the
        silhouette edge; the paint weight is 1.0 right at the edge and
        falls off quadratically to 0 at OUTER_PAD px out (or INNER_PAD px
        in), so the glow stays vivid at the outline but turns transparent
        quickly with distance. The transforms run at half resolution (the
        trail is blurred anyway, so the slight blockiness is invisible)
        for speed."""
        h, w = mask.shape
        small = cv2.resize(
            mask.astype(np.uint8), (w // 2, h // 2),
            interpolation=cv2.INTER_NEAREST,
        )
        dt_out = cv2.distanceTransform(
            (small == 0).astype(np.uint8), cv2.DIST_L2, 5
        ) * 2.0
        dt_in = cv2.distanceTransform(small, cv2.DIST_L2, 5) * 2.0
        strength = np.zeros((h // 2, w // 2), dtype=np.float32)
        out_rim = (small == 0) & (dt_out <= self.OUTER_PAD)
        in_rim = (small == 1) & (dt_in <= self.INNER_PAD)
        strength[out_rim] = 1.0 - dt_out[out_rim] / self.OUTER_PAD
        strength[in_rim] = 1.0 - dt_in[in_rim] / self.INNER_PAD
        strength *= strength                  # quadratic fade -> more transparent
        strength = cv2.resize(strength, (w, h), interpolation=cv2.INTER_NEAREST)
        band = strength > 0.0
        ys, xs = np.where(band)
        if ys.size == 0:
            return
        ys_c, xs_c = np.where(mask)
        cy, cx = float(np.mean(ys_c)), float(np.mean(xs_c))
        ang = np.arctan2(ys - cy, xs - cx)
        hue = self.HUE_BASE + ((np.degrees(ang) + t * self.HUE_SPEED) % self.HUE_RANGE)
        hsv = np.empty((ys.size, 1, 3), dtype=np.uint8)
        hsv[:, 0, 0] = (hue * 0.5).astype(np.uint8)
        hsv[:, 0, 1] = 255
        hsv[:, 0, 2] = 255
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3).astype(np.float32)
        wgt = (strength[ys, xs] * self.INTENSITY)[:, None]
        self.trail[ys, xs] += bgr * wgt

    def apply(self, frame, mask, t, dt=1.0 / 30.0, enabled=True):
        """Decay the trail, paint new color around ``mask`` (a frame-sized
        bool silhouette) if ``enabled``, blur it outward, and add it over
        the frame."""
        h, w = frame.shape[:2]
        self._ensure(h, w)
        self.trail *= self.DECAY
        if mask is not None and mask.any():
            self.trail[mask] *= self.MASK_DECAY
            if enabled:
                self._paint(mask, t)
        k = self.BLUR if self.BLUR % 2 == 1 else self.BLUR + 1
        self.trail = cv2.blur(self.trail, (k, k))
        out = frame.astype(np.float32) + self.trail
        return np.clip(out, 0, 255).astype(np.uint8)


class KaleidoscopeGestureState:
    """Pure gesture logic for the kaleidoscope toggle (no MediaPipe, no
    camera). Holding the prayer pose (both hands vertical, palms adjacent,
    thumbs touching) for HOLD_TIME toggles self.enabled on the rising edge
    (fires once until the pose is released)."""

    POINTS_UP = 1.0         # wrist->middle-tip over palm size; negative y is up
    PALM_DX = 0.12          # max normalized horizontal gap between palms
    PALM_DY = 0.18          # max normalized vertical gap between palms
    THUMB_DIST = 0.12       # max normalized distance between thumb tips
    HOLD_TIME = 0.6         # s the pose must be held to fire the toggle

    def __init__(self):
        self.enabled = False
        self.praying = False
        self.pose_start = None
        self.toggle_fired = False

    def _palm_center(self, landmarks):
        pts = [landmarks[i] for i in (0, 5, 17)]
        return np.array([np.mean([p.x for p in pts]), np.mean([p.y for p in pts])])

    def _points_up(self, landmarks):
        """Fingers pointing up (screen y grows downward)."""
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        middle_tip = landmarks[12]
        palm = max(
            np.linalg.norm([middle_mcp.x - wrist.x, middle_mcp.y - wrist.y]),
            1e-6,
        )
        d_y = (middle_tip.y - wrist.y) / palm
        return d_y < -self.POINTS_UP

    def detect_praying_pose(self, hands):
        """Both hands vertical, palms adjacent, thumbs touching."""
        if len(hands) < 2:
            return False
        a, b = hands[0], hands[1]
        if not (self._points_up(a) and self._points_up(b)):
            return False
        pa, pb = self._palm_center(a), self._palm_center(b)
        if abs(pa[0] - pb[0]) > self.PALM_DX or abs(pa[1] - pb[1]) > self.PALM_DY:
            return False
        thumb_a = np.array([a[4].x, a[4].y])
        thumb_b = np.array([b[4].x, b[4].y])
        return float(np.linalg.norm(thumb_a - thumb_b)) < self.THUMB_DIST

    def update(self, hands, now):
        """Advance the prayer-pose toggle state machine one frame. Returns
        True on the frame the toggle fires (rising edge)."""
        fired = False
        praying = self.detect_praying_pose(hands)
        if praying:
            if self.pose_start is None:
                self.pose_start = now
                self.toggle_fired = False
            elif not self.toggle_fired and now - self.pose_start >= self.HOLD_TIME:
                self.enabled = not self.enabled
                self.toggle_fired = True
                print(f"Kaleidoscope: {'ON' if self.enabled else 'OFF'}")
                fired = True
        else:
            self.pose_start = None
            self.toggle_fired = False
        self.praying = praying
        return fired


class WaveGestureState:
    """Pure gesture logic for the wave -> color-trail toggle (no MediaPipe,
    no camera). Waving both hands (oscillation of any of three signals:
    average x, average y, or palm separation) toggles self.wave_enabled
    once a completed wave is detected. See _track_axis for the hysteresis
    rationale."""

    WAVE_AMPLITUDE = 0.09   # normalized swing required to count a reversal
    WAVE_SWINGS = 3         # reversals to fire the toggle
    WAVE_WINDOW = 2.5       # s allowed for the full wave
    WAVE_TIMEOUT = 1.2      # s of stillness before the wave count resets
    WAVE_VEL_EPS = 0.02     # normalized speed below this counts as still
    WAVE_GRACE = 0.35       # s a hand may be lost before the wave resets
    WAVE_DEAD = 0.005       # per-frame delta dead-zone (kills micro-noise)

    def __init__(self):
        self.wave_enabled = False
        self.waving = False
        self.wave_samples = deque(maxlen=90)
        self.wave_swings = 0
        self.wave_start_t = None
        self.wave_last_t = None
        self.wave_drop_t = None
        self.wave_track = {
            "x": {"peak": None, "trough": None, "dir": None},
            "y": {"peak": None, "trough": None, "dir": None},
            "d": {"peak": None, "trough": None, "dir": None},
        }

    def _palm_center(self, landmarks):
        pts = [landmarks[i] for i in (0, 5, 17)]
        return np.array([np.mean([p.x for p in pts]), np.mean([p.y for p in pts])])

    def _wave_reset(self):
        self.wave_samples.clear()
        self.wave_swings = 0
        self.wave_start_t = None
        self.wave_last_t = None
        self.wave_drop_t = None
        for s in self.wave_track.values():
            s["peak"] = s["trough"] = s["prev"] = None
            s["dir"] = None

    def _track_axis(self, key, v, now):
        """Feed one scalar signal through turn detection.

        A turn is signalled when the per-frame delta flips direction past a
        dead-zone (so micro-noise is ignored), but the swing is only
        counted when the running peak-to-trough excursion since the last
        count reaches WAVE_AMPLITUDE. Sub-amplitude turns do NOT re-baseline
        the peak/trough, so jitter and small wiggles can't eat the
        amplitude of the real swing. This also works for rectified signals
        (like hand separation, which never goes below 0)."""
        s = self.wave_track[key]
        if s["peak"] is None:
            s["peak"] = s["trough"] = s["prev"] = v
            s["dir"] = None
            return
        if v > s["peak"]:
            s["peak"] = v
        if v < s["trough"]:
            s["trough"] = v
        delta = v - s["prev"]
        s["prev"] = v
        if abs(delta) < self.WAVE_DEAD:
            return                      # flat: keep current direction
        nd = "up" if delta > 0 else "down"
        if s["dir"] is not None and nd != s["dir"]:
            if s["peak"] - s["trough"] >= self.WAVE_AMPLITUDE:
                self.wave_swings += 1
                if self.wave_swings == 1:
                    self.wave_start_t = now
                self.wave_last_t = now
                s["peak"] = s["trough"] = v
        s["dir"] = nd

    def update(self, hands, now):
        """Detect waving (oscillation of the two hands) and toggle the
        color-diffusion effect once a wave completes. Returns True on the
        toggle frame."""
        if len(hands) < 2:
            self.waving = False
            # tolerate brief single-hand gaps without wiping the count
            if self.wave_samples:
                if self.wave_drop_t is None:
                    self.wave_drop_t = now
                elif now - self.wave_drop_t > self.WAVE_GRACE:
                    self._wave_reset()
            return False
        self.wave_drop_t = None

        palm_a = self._palm_center(hands[0])
        palm_b = self._palm_center(hands[1])
        avg = (palm_a + palm_b) / 2.0
        sep = float(np.linalg.norm(palm_a - palm_b))
        self.wave_samples.append((now, (float(avg[0]), float(avg[1]), sep)))

        moving = False
        if len(self.wave_samples) >= 2:
            (t0, p0), (t1, p1) = self.wave_samples[-2], self.wave_samples[-1]
            dt = t1 - t0
            if dt > 0:
                moving = (
                    np.linalg.norm(np.asarray(p1) - np.asarray(p0)) / dt
                    >= self.WAVE_VEL_EPS
                )

        self._track_axis("x", float(avg[0]), now)
        self._track_axis("y", float(avg[1]), now)
        self._track_axis("d", sep, now)
        self.waving = moving or self.wave_swings > 0

        if self.wave_swings > 0 and now - self.wave_last_t > self.WAVE_TIMEOUT:
            self._wave_reset()
            return False
        if (
            self.wave_swings >= self.WAVE_SWINGS
            and now - self.wave_start_t <= self.WAVE_WINDOW
        ):
            self.wave_enabled = not self.wave_enabled
            print(f"Color diffusion: {'ON' if self.wave_enabled else 'OFF'}")
            self._wave_reset()
            return True
        return False


class PersonTrackerState:
    """Pure person-box tracking logic (no camera): throttled segmentation
    via a PersonSegmenter, median-filtered over the last few updates (so a
    jumpy/outlier box can't yank the box) and low-pass eased toward,
    keeping the kaleidoscope anchor drift-free and smooth."""

    SEGMENT_EVERY = 3        # run person segmentation every Nth frame
    MAX_MISSED = 2           # segment frames without a person before resetting
    BOX_SMOOTH_WINDOW = 5    # recent boxes median-filtered before easing
    ANCHOR_SMOOTHING = 0.15  # low-pass factor for easing the bounding box

    def __init__(self, segmenter: PersonSegmenter):
        self.segmenter = segmenter
        self.frame_count = 0
        self.missed = 0
        self.box_history = deque(maxlen=self.BOX_SMOOTH_WINDOW)
        self.person_rect = None   # eased (x0, y0, x1, y1)
        self.center = None        # kaleidoscope anchor (person bbox center)

    def update(self, frame):
        """Run one throttled segmentation + smoothing step. Updates
        self.person_rect and self.center after each call."""
        self.frame_count += 1
        if self.frame_count % self.SEGMENT_EVERY != 0:
            return
        box = self.segmenter.bbox(frame)
        if box is None:
            self.missed += 1
            if self.missed > self.MAX_MISSED:
                self.person_rect = None
                self.center = None
                self.box_history.clear()
            return
        self.missed = 0
        self.box_history.append(box)
        target = tuple(np.median(self.box_history, axis=0))
        if self.person_rect is None:
            self.person_rect = [float(v) for v in target]
        else:
            a = self.ANCHOR_SMOOTHING
            self.person_rect = [
                self.person_rect[i] + (target[i] - self.person_rect[i]) * a
                for i in range(4)
            ]
        x0, y0, x1, y1 = self.person_rect
        self.center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


class GestureEffectsController:
    """Standalone camera controller: the prayer pose toggles the
    kaleidoscope, waving with both hands toggles the color-diffusion trail.
    Both effects stack when enabled together. All gesture logic is
    delegated to KaleidoscopeGestureState, WaveGestureState and
    PersonTrackerState."""

    def __init__(self, segments=Kaleidoscope.SEGMENTS):
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

        self.kaleido = Kaleidoscope(segments=segments)
        self.diffusion = ColorDiffusion()
        self.segmenter = PersonSegmenter()

        self.kaleido_state = KaleidoscopeGestureState()
        self.wave_state = WaveGestureState()
        self.person_tracker = PersonTrackerState(self.segmenter)

        self.frame_hands = []
        self.show_annotations = True

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

    def process_frame(self, frame, frame_timestamp_ms):
        """Detect hands once and delegate to the gesture state classes:
        prayer toggles the kaleidoscope on the rising edge (held for
        HOLD_TIME, fires once until released); a completed both-hands wave
        toggles the color-diffusion trail."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        self.frame_hands = [list(landmarks) for landmarks in result.hand_landmarks]

        now = frame_timestamp_ms / 1000.0
        self.kaleido_state.update(self.frame_hands, now)
        self.wave_state.update(self.frame_hands, now)

    def update_person(self, frame):
        """Throttled person segmentation feeding a smoothed bounding box.
        Delegated to PersonTrackerState."""
        self.person_tracker.update(frame)

    def run(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("Gesture Stream Effects Active!")
        print("- Hold your hands together as if praying to toggle the kaleidoscope")
        print("- Wave with both hands (side to side) to toggle the color trail")
        print("- 't' to manually toggle the color trail")
        print("- 'h' to toggle annotations (green = person mask, magenta = box)")
        print("- 'q' to quit")

        start_time = time.time()
        last_draw = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_timestamp_ms = int((time.time() - start_time) * 1000)
            self.process_frame(frame, frame_timestamp_ms)
            self.update_person(frame)

            now = time.time()
            dt = min(now - last_draw, 0.1)
            last_draw = now

            # both effects stack: kaleidoscope first (remaps the whole
            # frame), then the color trail paints on top of the result so
            # the glow stays visible over the reflections. The kaleidoscope
            # keeps rendering while its blend is still fading out so it
            # dissolves smoothly instead of snapping off.
            if self.kaleido_state.enabled or self.kaleido.blend > 0.01:
                if self.person_tracker.person_rect is not None:
                    frame = self.kaleido.apply(
                        frame, dt, box=self.person_tracker.person_rect,
                        active=self.kaleido_state.enabled,
                    )
                else:
                    frame = self.kaleido.apply(
                        frame, dt, active=self.kaleido_state.enabled
                    )

            if self.wave_state.wave_enabled or self.diffusion.is_active():
                frame = self.diffusion.apply(
                    frame,
                    self.segmenter.last_mask,
                    now,
                    dt,
                    enabled=self.wave_state.wave_enabled,
                )

            if self.show_annotations:
                h, w = frame.shape[:2]
                for landmarks in self.frame_hands:
                    for lm in landmarks:
                        cv2.circle(
                            frame,
                            (int(lm.x * w), int(lm.y * h)),
                            3,
                            (0, 255, 0),
                            -1,
                        )
                mask = self.segmenter.last_mask
                if mask is not None and mask.any():
                    # green tint over the segmented person + its contour
                    tint = np.zeros_like(frame)
                    tint[mask] = (0, 255, 0)
                    frame = cv2.addWeighted(frame, 1.0, tint, 0.4, 0)
                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)
                if self.person_tracker.person_rect is not None:
                    x0, y0, x1, y1 = [int(v) for v in self.person_tracker.person_rect]
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 255), 2)
                    cv2.circle(
                        frame,
                        (int(self.person_tracker.center[0]), int(self.person_tracker.center[1])),
                        4,
                        (255, 0, 255),
                        -1,
                    )

            state = "ON" if self.kaleido_state.enabled else "OFF"
            praying = "YES" if self.kaleido_state.praying else "NO"
            trail = "ON" if self.wave_state.wave_enabled else "OFF"
            waving = "YES" if self.wave_state.waving else "NO"
            person = "YES" if self.person_tracker.person_rect is not None else "NO"
            cv2.putText(
                frame,
                f"Kaleido: {state} | Trail: {trail} | Waving: {waving} | "
                f"Praying: {praying} | Person: {person}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 220, 255),
                2,
            )

            cv2.imshow("Gesture Stream Effects", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("h"):
                self.show_annotations = not self.show_annotations
                print(f"Annotations: {'ON' if self.show_annotations else 'OFF'}")
            elif key == ord("t"):
                self.wave_state.wave_enabled = not self.wave_state.wave_enabled
                print(f"Color diffusion: {'ON' if self.wave_state.wave_enabled else 'OFF'}")

        cap.release()
        cv2.destroyAllWindows()


def demo(frames=None):
    """Synthetic no-camera demo: a stylized person is drawn (swaying across
    the frame); the kaleidoscope and color trail toggle on/off on timers so
    both effects and their stacking can be seen."""
    print("Gesture Stream Effects Demo (No Camera Required)")
    print("Press 'q' to quit\n")

    h, w = 720, 1280
    background = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(w):
        background[:, i] = [int(i / w * 50), int(i / w * 100), 100]

    kaleido = Kaleidoscope()
    diffusion = ColorDiffusion()
    start = time.time()
    frame_count = 0

    while frames is None or frame_count < frames:
        t = time.time() - start
        cyc = t % 12.0
        kaleido_enabled = cyc < 6.0
        trail_enabled = 3.0 <= cyc < 9.0

        frame = background.copy()
        mask = np.zeros((h, w), dtype=np.uint8)

        # stylized person (head + torso + legs), slowly swaying; the same
        # shapes also fill a mask so the trail effect has a silhouette
        px = int(w / 2.0 + math.sin(t * 0.5) * w * 0.15)
        py = int(h * 0.42)
        body_w, body_h = 150, 280
        head_r = 45
        cv2.circle(
            frame,
            (px, py - body_h // 2 + 10),
            head_r,
            (120, 160, 220),
            -1,
        )
        cv2.circle(
            mask,
            (px, py - body_h // 2 + 10),
            head_r,
            255,
            -1,
        )
        cv2.rectangle(
            frame,
            (px - body_w // 2, py - body_h // 2 + 10 + head_r),
            (px + body_w // 2, py + body_h // 2),
            (160, 200, 240),
            -1,
        )
        cv2.rectangle(
            mask,
            (px - body_w // 2, py - body_h // 2 + 10 + head_r),
            (px + body_w // 2, py + body_h // 2),
            255,
            -1,
        )
        cv2.line(
            frame,
            (px - 40, py + body_h // 2),
            (px - 60, py + body_h // 2 + 60),
            (160, 200, 240),
            18,
        )
        cv2.line(
            mask,
            (px - 40, py + body_h // 2),
            (px - 60, py + body_h // 2 + 60),
            255,
            18,
        )
        cv2.line(
            frame,
            (px + 40, py + body_h // 2),
            (px + 60, py + body_h // 2 + 60),
            (160, 200, 240),
            18,
        )
        cv2.line(
            mask,
            (px + 40, py + body_h // 2),
            (px + 60, py + body_h // 2 + 60),
            255,
            18,
        )

        x0, y0 = px - body_w // 2, py - body_h // 2
        x1, y1 = px + body_w // 2, py + body_h // 2 + 60

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

        k_state = "ON" if kaleido_enabled else "OFF"
        t_state = "ON" if trail_enabled else "OFF"
        cv2.putText(
            frame,
            f"Kaleidoscope: {k_state} | Trail: {t_state}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 255),
            2,
        )

        cv2.imshow("Gesture Stream Effects Demo", frame)
        key = cv2.waitKey(33) & 0xFF
        if key == ord("q"):
            break
        frame_count += 1

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Gesture-controlled kaleidoscope + color-trail effects"
    )
    parser.add_argument("--demo", action="store_true", help="synthetic demo, no camera")
    args = parser.parse_args()

    if args.demo:
        demo()
    else:
        controller = GestureEffectsController()
        controller.run()


if __name__ == "__main__":
    main()
