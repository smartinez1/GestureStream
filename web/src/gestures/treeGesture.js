import { palmCenter, dist2d, cross, norm3, clip } from "./utils.js";
import { VoxelTree } from "../effects/tree.js";

export class TreeGestureState {
  static TOGETHER_DIST = 0.16;
  static SPREAD_DIST = 0.34;
  static TOGETHER_MIN = 0.2;
  static ANCHOR_SMOOTHING = 0.15;
  static ANCHOR_UP_OFFSET = 40;
  static PALM_UP_NORMAL_Y = 0.45;
  static PALM_UP_CONSECUTIVE = 3;

  constructor() {
    this.anchor = [0, 0];
    this._anchorSet = false;
    this.state = "idle";
    this.togetherSince = null;
    this.spreadPeak = 0.0;
    this.palmsUp = false;
    this._palmUpFrames = 0;
  }

  _palmNormal(lm) {
    const wrist = [lm[0].x, lm[0].y, lm[0].z];
    const middleMcp = [lm[9].x, lm[9].y, lm[9].z];
    const indexMcp = [lm[5].x, lm[5].y, lm[5].z];
    const pinkyMcp = [lm[17].x, lm[17].y, lm[17].z];
    const n = cross(
      [middleMcp[0] - wrist[0], middleMcp[1] - wrist[1], middleMcp[2] - wrist[2]],
      [pinkyMcp[0] - indexMcp[0], pinkyMcp[1] - indexMcp[1], pinkyMcp[2] - indexMcp[2]]
    );
    const len = norm3(n);
    if (len < 1e-6) return null;
    return [n[0] / len, n[1] / len, n[2] / len];
  }

  _isPalmUp(lm, handedness) {
    const normal = this._palmNormal(lm);
    if (normal === null) return false;
    if (handedness === "Right") return normal[1] < -TreeGestureState.PALM_UP_NORMAL_Y;
    return normal[1] > TreeGestureState.PALM_UP_NORMAL_Y;
  }

  update(hands, handedness, frameSize, now, tree) {
    const [h, w] = frameSize;
    let both = hands.length >= 2;
    let d = null;
    if (both) {
      const palms = hands.map((lm) => palmCenter(lm));
      d = dist2d(palms[0], palms[1]);
      const target = [
        Math.floor((palms[0].x * w + palms[1].x * w) / 2),
        Math.floor((palms[0].y * h + palms[1].y * h) / 2),
      ];
      if (!this._anchorSet) {
        this.anchor = target;
        this._anchorSet = true;
      } else {
        const a = TreeGestureState.ANCHOR_SMOOTHING;
        this.anchor = [
          Math.round(this.anchor[0] + (target[0] - this.anchor[0]) * a),
          Math.round(this.anchor[1] + (target[1] - this.anchor[1]) * a),
        ];
      }
      const labels = handedness.length >= 2 ? handedness : ["Left", "Left"];
      both = this._isPalmUp(hands[0], labels[0]) && this._isPalmUp(hands[1], labels[1]);
    } else {
      this._anchorSet = false;
    }
    if (both) this._palmUpFrames += 1;
    else this._palmUpFrames = 0;
    this.palmsUp = this._palmUpFrames >= TreeGestureState.PALM_UP_CONSECUTIVE;

    const twoHands = hands.length >= 2;
    if (this.state === "idle") {
      if (twoHands && d > TreeGestureState.SPREAD_DIST) {
        this.state = "spread";
        this.spreadPeak = d;
      }
    } else if (this.state === "spread") {
      if (!twoHands) {
        if (tree.isActive() && !tree.fading) tree.startFade();
        this.state = "idle";
      } else if (d > TreeGestureState.SPREAD_DIST) {
        this.spreadPeak = Math.max(this.spreadPeak, d);
      } else if (d < TreeGestureState.TOGETHER_DIST && this.palmsUp) {
        this.state = "together";
        this.togetherSince = now;
      }
    } else if (this.state === "together") {
      if (!twoHands) {
        if (tree.isActive() && !tree.fading) tree.startFade();
        this.state = "idle";
      } else if (d > TreeGestureState.SPREAD_DIST) {
        if (tree.isActive() && !tree.fading) tree.startFade();
        this.state = "spread";
        this.spreadPeak = d;
      } else if (d < TreeGestureState.TOGETHER_DIST) {
        if (!this.palmsUp) {
          this.togetherSince = null;
        } else if (this.togetherSince === null) {
          this.togetherSince = now;
        } else if (
          now - this.togetherSince >= TreeGestureState.TOGETHER_MIN &&
          !tree.isActive()
        ) {
          const scale = clip(
            this.spreadPeak / TreeGestureState.SPREAD_DIST,
            0.9,
            VoxelTree.MAX_HEIGHT_SCALE
          );
          tree.spawn(scale);
        }
      } else {
        this.togetherSince = null;
      }
    }
  }
}