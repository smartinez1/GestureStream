"""
Gesture Stream Tree - stylized 3D tree that grows from the palms.

Gesture (two-hand "U" squeeze, palms up):
    1. START  - both hands held apart (arms the gesture).
    2. GROW   - palms-up hands come together: a tree grows between the
                palms, scaled by how wide the hands were spread. The tree
                tracks the hands' midpoint while they move. Only a
                palms-up squeeze counts, so a vertical prayer pose never
                grows a tree.
    3. FINAL  - the together pose is the final pose; once it finishes
                (hands spread apart or drop), the tree fades out and
                disappears.

Self-contained: does not import any other gesture_stream module. Uses the
MediaPipe Task API directly (VIDEO mode, 2 hands) and renders a procedural
3D tree (recursive branching, tapered curved trunk, organic leaf canopy;
orthographic projection, rz@ry@rx convention like the core cube) with a
growth animation and a reverse fade-out with falling leaves.

Usage:
    python3 gesture_stream_tree.py            # camera
    python3 gesture_stream_tree.py --demo     # synthetic demo, no camera
"""

import argparse
import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np


class VoxelTree:
    """Procedural 3D tree with realistic recursive branching.

    The model (a curved tapered trunk, two generations of branches and a
    dense organic canopy) is generated once per spawn, then animated with a
    single progress value:
        grow: the trunk rises, branches extend outward from the trunk,
              then the canopy leaves unfurl;
        fade: leaves fall while everything fades away and disappears.
    Wind sways the whole tree, stronger higher up.
    """

    TRUNK_HEIGHT = 180.0         # px at height_scale 1.0
    TRUNK_RADIUS = 22.0
    SEG_LEN = 15.0               # px target length of one branch segment
    BRANCHES_L1 = 5              # main branches from the trunk
    BRANCHES_L2 = 3              # sub-branches per main branch
    L2_LENGTH = 0.55             # sub-branch length as a fraction of L1
    LEAF_TWIG = 7                # leaves clustered at each twig tip
    LEAF_CROWN = 90              # leaves filling the upper canopy
    GROW_DURATION = 3.0          # s
    FADE_DURATION = 2.0          # s
    MAX_HEIGHT_SCALE = 1.6
    FALL_DIST = 60.0             # px a leaf drops during the fade
    SWAY = 0.7                   # px lateral sway amplitude
    SWAY_WIND = 1.7              # rad/s sway frequency

    BARK_BOTTOM = (35, 60, 90)   # BGR dark brown at the base
    BARK_TOP = (60, 105, 150)    # lighter brown near the crown
    BARK_L1 = (50, 85, 120)      # main branch bark
    BARK_L2 = (58, 100, 140)     # twig bark
    LEAF_GREENS = [              # BGR, dark (shaded, low) -> bright (lit, top)
        (50, 130, 65),
        (70, 160, 80),
        (95, 190, 100),
        (120, 205, 110),
        (150, 215, 120),
        (175, 225, 135),
    ]
    LEAF_ACCENTS = [             # occasional warm/autumn leaves
        (0, 120, 190),
        (165, 135, 55),
        (200, 165, 90),
    ]

    def __init__(self, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.segments = []
        self.leaves = []
        self.active = False
        self.growing = False
        self.fading = False
        self.grow_time = 0.0
        self.fade_time = 0.0
        self._trunk_h = self.TRUNK_HEIGHT

    # ---- lifecycle -----------------------------------------------------

    def spawn(self, height_scale=1.0):
        """Build a fresh tree model and start the growth animation."""
        self.segments, self.leaves = self._build(height_scale)
        self._trunk_h = self.TRUNK_HEIGHT * height_scale
        self.active = True
        self.growing = True
        self.fading = False
        self.grow_time = 0.0
        self.fade_time = 0.0

    def start_fade(self):
        """Begin the disappearance animation (no-op if already fading)."""
        if self.active and not self.fading:
            self.fading = True
            self.growing = False
            self.fade_time = 0.0

    def is_active(self):
        return self.active

    # ---- model generation ---------------------------------------------

    def _lerp_color(self, c1, c2, f):
        return tuple(int(a + (b - a) * f) for a, b in zip(c1, c2))

    def _jitter(self, color, amount):
        return tuple(
            int(np.clip(c + self.rng.uniform(-amount, amount), 0, 255))
            for c in color
        )

    def _point_on_trunk(self, segments, f):
        """3D position at fraction f (0 base -> 1 top) along the trunk."""
        trunk = [s for s in segments if s["gen"] == 0]
        n = len(trunk)
        if n == 0:
            return np.zeros(3)
        f = float(np.clip(f, 0.0, 0.999))
        idx = int(f * n)
        t = f * n - idx
        seg = trunk[idx]
        return seg["a"] * (1.0 - t) + seg["b"] * t

    def _branch(self, segments, origin, direction, length, radius, gen,
                reveal_base, height_scale):
        """Walk a branch outward, adding tapered segments that arc upward
        with organic jitter. Returns the final (position, direction)."""
        rng = self.rng
        seg_len = self.SEG_LEN * height_scale
        n_seg = max(3, int(length / seg_len))
        pos = origin.copy()
        d = direction.copy()
        bend = 0.06 if gen == 1 else 0.10
        for i in range(n_seg):
            f = i / n_seg
            d[1] += bend                       # branches arc upward
            d[1] = max(d[1], 0.12)
            d += rng.normal(0.0, 0.05 * (gen + 1), 3)
            d /= np.linalg.norm(d)
            end = pos + d * (length / n_seg)
            r = radius * (1.0 - 0.65 * f)
            color = self.BARK_L1 if gen == 1 else self.BARK_L2
            segments.append(
                {
                    "a": pos.copy(), "b": end,
                    "r": max(r, 0.8), "gen": gen,
                    "color": color,
                    "reveal": reveal_base + f * 0.20,
                    "phase": rng.uniform(0.0, 2.0 * np.pi),
                }
            )
            pos = end
        return pos, d

    def _leaf(self, pos, radius, reveal):
        rng = self.rng
        hf = float(np.clip(pos[1] / self.TRUNK_HEIGHT + 0.5, 0.0, 1.0))
        if rng.random() < 0.10:
            color = self.LEAF_ACCENTS[
                int(rng.integers(0, len(self.LEAF_ACCENTS)))
            ]
        else:
            base = int(np.clip(hf, 0.0, 1.0) * (len(self.LEAF_GREENS) - 1))
            color = self._jitter(self.LEAF_GREENS[base], 12)
        return {
            "pos": np.array(pos, dtype=np.float32),
            "r": float(radius),
            "color": color,
            "reveal": float(reveal),
            "phase": rng.uniform(0.0, 2.0 * np.pi),
        }

    def _build(self, height_scale):
        segments = []
        leaves = []
        rng = self.rng
        trunk_h = self.TRUNK_HEIGHT * height_scale
        trunk_r = self.TRUNK_RADIUS * height_scale

        # --- trunk: slightly curved, flared at the base ---
        pos = np.array([0.0, -trunk_h / 2, 0.0], dtype=np.float32)
        lean = rng.uniform(-0.12, 0.12)
        d = np.array([lean, 1.0, rng.uniform(-0.08, 0.08)], dtype=np.float32)
        d /= np.linalg.norm(d)
        trunk_len = trunk_h * 0.52
        n_seg = max(4, int(trunk_len / (self.SEG_LEN * height_scale)))
        for i in range(n_seg):
            f = i / n_seg
            d[0] += (lean * 0.5 - d[0]) * 0.15     # ease back toward the lean
            d[1] += 0.03
            d += rng.normal(0.0, 0.015, 3)
            d /= np.linalg.norm(d)
            end = pos + d * (trunk_len / n_seg)
            r = trunk_r * (1.0 - 0.72 * f)
            if i == 0:
                r = trunk_r * 1.2                  # root flare
            color = self._jitter(
                self._lerp_color(self.BARK_BOTTOM, self.BARK_TOP, f), 6
            )
            segments.append(
                {
                    "a": pos.copy(), "b": end,
                    "r": max(r, 1.2), "gen": 0,
                    "color": color,
                    "reveal": f * 0.22,
                    "phase": rng.uniform(0.0, 2.0 * np.pi),
                }
            )
            pos = end

        # --- level 1 branches from along the upper trunk ---
        for b in range(self.BRANCHES_L1):
            f = rng.uniform(0.45, 0.95)
            origin = self._point_on_trunk(segments, f)
            az = rng.uniform(0.0, 2.0 * np.pi)
            bd = np.array(
                [np.cos(az), rng.uniform(0.5, 0.85), np.sin(az)],
                dtype=np.float32,
            )
            bd[1] = abs(bd[1])
            bd /= np.linalg.norm(bd)
            length = trunk_h * rng.uniform(0.30, 0.42)
            r = trunk_r * rng.uniform(0.20, 0.28)
            end1, bd = self._branch(
                segments, origin, bd, length, r, 1, 0.24, height_scale
            )

            # leaves scattered along the main branch
            for _ in range(6):
                t = rng.uniform(0.5, 1.0)
                lpos = origin * (1.0 - t) + end1 * t
                leaves.append(
                    self._leaf(lpos, rng.uniform(2.0, 3.2), 0.78 + rng.uniform(0.0, 0.18))
                )

            # level 2 sub-branches -> leafy twigs
            for _ in range(self.BRANCHES_L2):
                sd = bd + rng.normal(0.0, 0.25, 3)
                sd[1] += 0.25
                sd /= np.linalg.norm(sd)
                sub_len = length * self.L2_LENGTH * rng.uniform(0.8, 1.2)
                end2, _ = self._branch(
                    segments, end1, sd, sub_len, r * 0.55, 2, 0.52, height_scale
                )
                for _ in range(self.LEAF_TWIG):
                    off = rng.normal(0.0, 1.0, 3)
                    off = off / (np.linalg.norm(off) + 1e-9) * rng.uniform(
                        0.0, trunk_r * 1.1 * height_scale
                    )
                    leaves.append(
                        self._leaf(end2 + off, rng.uniform(2.2, 3.6), 0.86 + rng.uniform(0.0, 0.14))
                    )

        # --- fill the canopy: leaves through the upper crown ---
        crown_c = np.array([lean * trunk_h * 0.08, trunk_h * 0.30, 0.0])
        crown_rx = trunk_h * 0.34
        crown_ry = trunk_h * 0.28
        for _ in range(self.LEAF_CROWN):
            p = rng.normal(0.0, 1.0, 3)
            p = p / (np.linalg.norm(p) + 1e-9) * rng.uniform(0.0, 1.0)
            lpos = crown_c + p * np.array([crown_rx, crown_ry, crown_rx])
            if lpos[1] < 0:
                lpos[1] = -lpos[1] * 0.5
            leaves.append(
                self._leaf(lpos, rng.uniform(2.2, 3.6), 0.80 + rng.uniform(0.0, 0.2))
            )

        return segments, leaves

    # ---- rendering -----------------------------------------------------

    def _smoothstep(self, t):
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def _rotation_matrix(rotation):
        rx, ry, rz = rotation
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return rot_z @ rot_y @ rot_x

    def _blend_line(self, frame, pt1, pt2, thickness, color, alpha):
        """Draw a line, alpha-blending over the background when translucent.

        The primitive is drawn onto a black buffer (LINE_AA gives a color
        premultiplied by per-pixel coverage), then composited with the same
        coverage so the anti-aliased edges don't darken the background.
        """
        if alpha >= 0.98:
            cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
            return
        h, w = frame.shape[:2]
        r = thickness + 1
        x0 = max(0, min(pt1[0], pt2[0]) - r)
        x1 = min(w, max(pt1[0], pt2[0]) + r)
        y0 = max(0, min(pt1[1], pt2[1]) - r)
        y1 = min(h, max(pt1[1], pt2[1]) + r)
        if x1 <= x0 or y1 <= y0:
            return
        sub = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
        cv2.line(
            sub,
            (pt1[0] - x0, pt1[1] - y0),
            (pt2[0] - x0, pt2[1] - y0),
            color, thickness, cv2.LINE_AA,
        )
        cov = np.clip((sub / np.maximum(np.array(color, np.float32), 1.0)).max(axis=2), 0.0, 1.0)
        a = (alpha * cov)[..., None]
        roi = frame[y0:y1, x0:x1].astype(np.float32)
        frame[y0:y1, x0:x1] = (roi * (1.0 - a) + sub * alpha).astype(np.uint8)

    def _blend_circle(self, frame, center, radius, color, alpha):
        if alpha >= 0.98:
            cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)
            return
        h, w = frame.shape[:2]
        r = radius + 1
        x0, y0 = center[0] - r, center[1] - r
        x1, y1 = center[0] + r, center[1] + r
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(w, x1), min(h, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        sub = np.zeros((cy1 - cy0, cx1 - cx0, 3), np.float32)
        cv2.circle(
            sub,
            (center[0] - cx0, center[1] - cy0),
            radius, color, -1, cv2.LINE_AA,
        )
        cov = np.clip((sub / np.maximum(np.array(color, np.float32), 1.0)).max(axis=2), 0.0, 1.0)
        a = (alpha * cov)[..., None]
        roi = frame[cy0:cy1, cx0:cx1].astype(np.float32)
        frame[cy0:cy1, cx0:cx1] = (roi * (1.0 - a) + sub * alpha).astype(np.uint8)

    def draw(self, frame, base, rotation, dt=1.0 / 30.0):
        """Advance the animation and draw the tree, orthographically
        projected and depth-sorted (far elements first)."""
        if not self.active:
            return frame

        grow_p = min(self.grow_time / self.GROW_DURATION, 1.0)
        fade_p = 0.0
        if self.growing:
            self.grow_time += dt
            grow_p = min(self.grow_time / self.GROW_DURATION, 1.0)
            if grow_p >= 1.0:
                self.growing = False
        elif self.fading:
            self.fade_time += dt
            fade_p = min(self.fade_time / self.FADE_DURATION, 1.0)
            if fade_p >= 1.0:
                self.active = False
                return frame
        # else: fully grown, held until start_fade() is called.

        t = time.time()
        rot = self._rotation_matrix(rotation)
        shift = self._trunk_h / 2          # local y=-h/2 (roots) lands on base
        h, w = frame.shape[:2]
        items = []

        # branches: tapered lines that extend outward as they grow
        for seg in self.segments:
            a = seg["a"].copy()
            b = seg["b"].copy()
            height_f = float(
                np.clip((a[1] + b[1]) / 2 / self.TRUNK_HEIGHT + 0.5, 0.0, 1.2)
            )
            sway = self.SWAY * height_f * np.sin(t * self.SWAY_WIND + seg["phase"])
            a[0] += sway
            b[0] += sway

            ra = rot @ a
            rb = rot @ b
            tf = (grow_p - seg["reveal"]) / 0.05
            if tf <= 0.0:
                continue
            tf = min(tf, 1.0)
            rb_eff = ra + (rb - ra) * tf

            ax = base[0] + ra[0]
            ay = base[1] - (ra[1] + shift)
            bx = base[0] + rb_eff[0]
            by = base[1] - (rb_eff[1] + shift)
            r = max(1, int(2 * seg["r"]))
            if (
                (ax < -r and bx < -r) or (ax > w + r and bx > w + r)
                or (ay < -r and by < -r) or (ay > h + r and by > h + r)
            ):
                continue
            alpha = 1.0 - fade_p
            items.append(
                (
                    (ra[2] + rb_eff[2]) * 0.5,
                    "seg",
                    (ax, ay, bx, by, r, seg["color"], alpha),
                )
            )

        # leaves: small circles that unfurl, sway, and fall on fade
        for leaf in self.leaves:
            pos = leaf["pos"].copy()
            height_f = float(np.clip(pos[1] / self.TRUNK_HEIGHT + 0.5, 0.0, 1.2))
            if fade_p > 0.0:
                pos[1] -= fade_p * self.FALL_DIST * (0.5 + 0.5 * leaf["phase"])
                pos[0] += np.sin(t * 2.0 + leaf["phase"]) * fade_p * 3.0
            sway = self.SWAY * height_f * np.sin(t * self.SWAY_WIND + leaf["phase"])
            pos[0] += sway + np.sin(t * 3.1 + leaf["phase"]) * 0.6

            rp = rot @ pos
            sx = base[0] + rp[0]
            sy = base[1] - (rp[1] + shift)
            if sx < -10 or sx > w + 10 or sy < -10 or sy > h + 10:
                continue
            alpha = self._smoothstep((grow_p - leaf["reveal"]) / 0.07)
            alpha *= (1.0 - fade_p)
            if alpha <= 0.0:
                continue
            items.append((rp[2], "leaf", (sx, sy, leaf["r"], leaf["color"], alpha)))

        items.sort(key=lambda it: it[0])          # far (larger z) first
        for _, kind, data in items:
            if kind == "seg":
                x1, y1, x2, y2, r, color, alpha = data
                self._blend_line(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), r, color, alpha
                )
            else:
                sx, sy, r, color, alpha = data
                self._blend_circle(
                    frame, (int(sx), int(sy)), max(1, int(r)), color, alpha
                )
        return frame


class TreeGestureState:
    """Pure gesture logic for the tree effect (no MediaPipe, no camera).

    Drives the spread -> together state machine: both hands apart arms the
    gesture, bringing them together while BOTH PALMS FACE UP and holding
    TOGETHER_MIN seconds spawns the tree (scaled by the widest spread);
    spreading apart again or dropping the hands fades it. The palms-up
    requirement gates only the squeeze and the hold, so a vertical prayer
    pose (palms facing each other) arms but never spawns the tree. The
    anchor tracks the palms' midpoint, low-pass eased so hand jitter
    doesn't vibrate the tree. Calls tree.spawn() / tree.start_fade()
    directly."""

    TOGETHER_DIST = 0.16        # normalized inter-palm distance: "together"
    SPREAD_DIST = 0.34          # crossing this = "spread apart"
    TOGETHER_MIN = 0.2          # s hands must stay together to grow
    ANCHOR_SMOOTHING = 0.15     # per-frame ease of the anchor toward the palms
    ANCHOR_UP_OFFSET = 40       # px the tree base sits above the palms
    PALM_UP_NORMAL_Y = 0.45     # palm normal must point this far up (see _is_palm_up)
    PALM_UP_CONSECUTIVE = 3     # consecutive frames both hands palm-up to count

    def __init__(self):
        self.anchor = (0, 0)
        self._anchor_set = False
        self.state = "idle"
        self.together_since = None
        self.spread_peak = 0.0     # widest hand spread while arming
        self._palm_pts = []       # (x, y) palm centers this frame
        self._dist = None         # inter-palm distance this frame
        self.palms_up = False     # both hands palm-up (smoothed over frames)
        self._palm_up_frames = 0

    def _palm_center(self, landmarks):
        pts = [landmarks[i] for i in (0, 5, 17)]
        return np.array([np.mean([p.x for p in pts]), np.mean([p.y for p in pts])])

    def _palm_normal(self, landmarks):
        """Unit palm-plane normal: cross product of the wrist->middle-MCP
        and index-MCP->pinky-MCP vectors (landmark z included)."""
        wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
        middle_mcp = np.array([landmarks[9].x, landmarks[9].y, landmarks[9].z])
        index_mcp = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
        pinky_mcp = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])
        normal = np.cross(middle_mcp - wrist, pinky_mcp - index_mcp)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            return None
        return normal / norm

    def _is_palm_up(self, landmarks, handedness):
        """Palm facing up (fingers toward the camera): the palm normal must
        point up, which flips sign with the hand's handedness. Calibrated
        on the real camera (mirrored pipeline): a hand the landmarker
        labels "Left" gives normal_y ~ +1 when palm-up, "Right" gives ~ -1
        (image y grows downward)."""
        normal = self._palm_normal(landmarks)
        if normal is None:
            return False
        if handedness == "Right":
            return normal[1] < -self.PALM_UP_NORMAL_Y
        return normal[1] > self.PALM_UP_NORMAL_Y

    def update(self, hands, handedness, frame_size, now, tree):
        """Advance the spread->together state machine one frame.

        ``hands`` are the current frame's hand-landmark lists,
        ``handedness`` the parallel per-hand labels ("Left"/"Right") from
        the hand landmarker, ``now`` is the seconds timestamp, ``tree`` is
        the VoxelTree instance being driven (spawned/faded here). Updates
        self.anchor, self._palm_pts, self._dist, self.palms_up and
        self.state."""
        h, w = frame_size
        self._palm_pts = []
        self._dist = None

        if len(hands) >= 2:
            palms = [self._palm_center(lm) for lm in hands]
            self._palm_pts = [
                (int(p[0] * w), int(p[1] * h)) for p in palms
            ]
            self._dist = float(np.linalg.norm(palms[0] - palms[1]))
            # the tree follows the hands: anchor at the palms' midpoint,
            # eased toward it so hand jitter doesn't vibrate the tree
            target = (
                (self._palm_pts[0][0] + self._palm_pts[1][0]) // 2,
                (self._palm_pts[0][1] + self._palm_pts[1][1]) // 2,
            )
            if not self._anchor_set:
                self.anchor = target
                self._anchor_set = True
            else:
                self.anchor = (
                    int(self.anchor[0] + (target[0] - self.anchor[0]) * self.ANCHOR_SMOOTHING),
                    int(self.anchor[1] + (target[1] - self.anchor[1]) * self.ANCHOR_SMOOTHING),
                )
            labels = handedness if len(handedness) >= 2 else ["Left", "Left"]
            both_up = self._is_palm_up(hands[0], labels[0]) and self._is_palm_up(
                hands[1], labels[1]
            )
        else:
            both_up = False
            self._anchor_set = False
        if both_up:
            self._palm_up_frames += 1
        else:
            self._palm_up_frames = 0
        self.palms_up = self._palm_up_frames >= self.PALM_UP_CONSECUTIVE

        d = self._dist
        both = len(hands) >= 2

        if self.state == "idle":
            if both and d > self.SPREAD_DIST:
                self.state = "spread"
                self.spread_peak = d

        elif self.state == "spread":
            if not both:
                if tree.is_active() and not tree.fading:
                    tree.start_fade()
                self.state = "idle"
            elif d > self.SPREAD_DIST:
                self.spread_peak = max(self.spread_peak, d)
            elif d < self.TOGETHER_DIST and self.palms_up:
                self.state = "together"
                self.together_since = now

        elif self.state == "together":
            if not both:
                if tree.is_active() and not tree.fading:
                    tree.start_fade()
                self.state = "idle"
            elif d > self.SPREAD_DIST:
                # the final (together) pose finished -> tree disappears
                if tree.is_active() and not tree.fading:
                    tree.start_fade()
                    print("Tree fading")
                self.state = "spread"
                self.spread_peak = d
            elif d < self.TOGETHER_DIST:
                if not self.palms_up:
                    # palms rotated away mid-hold: restart the hold timer
                    self.together_since = None
                elif self.together_since is None:
                    self.together_since = now
                elif (
                    now - self.together_since >= self.TOGETHER_MIN
                    and not tree.is_active()
                ):
                    scale = float(
                        np.clip(
                            self.spread_peak / self.SPREAD_DIST,
                            0.9,
                            VoxelTree.MAX_HEIGHT_SCALE,
                        )
                    )
                    tree.spawn(height_scale=scale)
                    print(f"Tree spawned (scale {scale:.2f})")
            else:
                # drifted back apart mid-hold: restart the hold timer
                self.together_since = None


class TreeGestureController:
    """Standalone camera controller: hands spread -> together with palms
    up grows the tree between the palms (which follows the hands);
    spreading apart again (or dropping them) fades it. Gesture logic
    delegated to TreeGestureState."""

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

        self.tree = VoxelTree()
        self.tree_gesture = TreeGestureState()
        self.frame_hands = []
        self.show_annotations = True

    def _get_model_path(self):
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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        self.frame_hands = [list(landmarks) for landmarks in result.hand_landmarks]
        handedness = [h[0].category_name for h in result.handedness]
        self.tree_gesture.update(
            self.frame_hands,
            handedness,
            frame.shape[:2],
            frame_timestamp_ms / 1000.0,
            self.tree,
        )

    def run(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("Gesture Stream Tree Active!")
        print("- Hold your hands apart, then bring them together (palms up):")
        print("  a tree grows")
        print("- The tree follows your hands; spread them apart to fade it")
        print("- 'h' to toggle hand annotations")
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

            now = time.time()
            dt = min(now - last_draw, 0.1)
            last_draw = now

            if self.tree.is_active():
                t = time.time() - start_time
                rotation = np.array(
                    [
                        0.22 * np.sin(t * 0.7),
                        0.30 * np.cos(t * 0.5),
                        0.15 * np.sin(t * 0.9),
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

            if self.tree.is_active():
                if self.tree.growing:
                    status = "GROWING"
                elif self.tree.fading:
                    status = "FADING"
                else:
                    status = "HELD"
            else:
                status = "IDLE"
            cv2.putText(
                frame,
                f"Tree: {status} | State: {self.tree_gesture.state.upper()} | "
                f"Palms up: {'YES' if self.tree_gesture.palms_up else 'NO'}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 220, 255),
                2,
            )

            cv2.imshow("Gesture Stream Tree", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("h"):
                self.show_annotations = not self.show_annotations
                print(f"Annotations: {'ON' if self.show_annotations else 'OFF'}")

        cap.release()
        cv2.destroyAllWindows()


def demo(frames=None):
    """Synthetic no-camera demo driving the tree through the full cycle:
    palms spread apart -> come together (tree grows, following the palms)
    -> hold -> spread apart again (tree fades)."""
    print("Gesture Stream Tree Demo (No Camera Required)")
    print("Press 'q' to quit\n")

    h, w = 720, 1280
    background = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(w):
        background[:, i] = [int(i / w * 50), int(i / w * 100), 100]

    tree = VoxelTree()
    center = (w // 2, h // 2 + 40)
    start = time.time()
    frame_count = 0
    phase = "spread"
    anchor = center

    while frames is None or frame_count < frames:
        t = time.time() - start
        cyc = t % 13.0

        # palms trace the gesture: spread apart -> together (tree grows,
        # anchored at their midpoint) -> spread apart again (tree fades).
        if cyc < 3.0:                    # spread apart (arming)
            px1, px2 = center[0] - 450, center[0] + 450
            phase = "spread"
        elif cyc < 5.0:                  # coming together
            f = (cyc - 3.0) / 2.0
            px1 = int(center[0] - 450 + f * 420)
            px2 = int(center[0] + 450 - f * 420)
            phase = "together"
        elif cyc < 8.0:                  # held together (tree grows)
            px1, px2 = center[0] - 30, center[0] + 30
            phase = "held"
        elif cyc < 10.0:                 # final pose finishes: spread apart
            f = (cyc - 8.0) / 2.0
            px1 = int(center[0] - 30 - f * 420)
            px2 = int(center[0] + 30 + f * 420)
            phase = "release"
        else:                            # released
            px1, px2 = center[0] - 450, center[0] + 450
            phase = "released"

        py = center[1]
        if phase in ("together", "held"):
            py = center[1] + 30

        # the tree follows the palms' midpoint
        anchor = ((px1 + px2) // 2, py)

        if cyc >= 5.0 and cyc < 8.0 and not tree.is_active():
            tree.spawn(height_scale=1.0)
        elif cyc >= 8.0 and cyc < 13.0 and tree.is_active() and not tree.fading:
            tree.start_fade()

        frame = background.copy()
        rotation = np.array(
            [
                0.22 * np.sin(t * 0.7),
                0.30 * np.cos(t * 0.5),
                0.15 * np.sin(t * 0.9),
            ]
        )
        frame = tree.draw(
            frame,
            (anchor[0], anchor[1] - 40),
            rotation,
            dt=1.0 / 30.0,
        )

        # draw the two palms following the gesture path
        for x in (px1, px2):
            cv2.circle(frame, (x, py), 38, (90, 90, 160), 3)
            cv2.circle(frame, (x, py), 30, (40, 40, 120), -1)

        status = "IDLE"
        if tree.is_active():
            if tree.growing:
                status = "GROWING"
            elif tree.fading:
                status = "FADING"
            else:
                status = "HELD"
        cv2.putText(
            frame,
            f"Phase: {phase.upper()} | Tree: {status}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 255),
            2,
        )

        cv2.imshow("Gesture Stream Tree Demo", frame)
        key = cv2.waitKey(33) & 0xFF
        if key == ord("q"):
            break
        frame_count += 1

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Gesture-controlled procedural tree")
    parser.add_argument("--demo", action="store_true", help="synthetic demo, no camera")
    args = parser.parse_args()

    if args.demo:
        demo()
    else:
        controller = TreeGestureController()
        controller.run()


if __name__ == "__main__":
    main()