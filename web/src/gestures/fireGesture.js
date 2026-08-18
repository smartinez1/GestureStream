import { palmCenter, dist2d, clip } from "./utils.js";

export class FireGestureState {
  static HEAT_SMOOTHING = 0.12;
  static BURN_SMOOTHING = 0.12;
  static L_GRACE_FRAMES = 6;
  static L_INDEX_OFFSET = 0.05;
  static L_CURL_OFFSET = 0.02;
  static L_THUMB_MIN_RATIO = 1.15;

  constructor() {
    this.triggerCenter = null;
    this.heat = 0.05;
    this.flame = 0.0;
    this.burning = false;
    this._missFrames = 0;
    this._heldTarget = 0.0;
  }

  _isLSign(lm) {
    const indexUp = lm[8].y < lm[6].y - FireGestureState.L_INDEX_OFFSET;
    if (!indexUp) return false;
    for (const [tip, pip] of [[12, 10], [16, 14], [20, 18]]) {
      if (lm[tip].y < lm[pip].y - FireGestureState.L_CURL_OFFSET) return false;
    }
    return (
      dist2d(lm[4], lm[0]) / Math.max(dist2d(lm[3], lm[0]), 1e-6) >=
      FireGestureState.L_THUMB_MIN_RATIO
    );
  }

  update(hands, frameSize) {
    const [h, w] = frameSize;
    let trigger = null;
    let modulator = null;
    for (const lm of hands) {
      if (trigger === null && this._isLSign(lm)) trigger = lm;
      else if (modulator === null) modulator = lm;
    }
    let target;
    if (trigger !== null) {
      this._missFrames = 0;
      const pc = palmCenter(trigger);
      this.triggerCenter = [Math.round(pc.x * w), Math.round(pc.y * h)];
      if (modulator !== null) {
        const modPc = palmCenter(modulator);
        target = clip(1.0 - modPc.y, 0, 1);
      } else {
        target = 0.0;
      }
      this._heldTarget = target;
    } else {
      this._missFrames += 1;
      if (this._missFrames <= FireGestureState.L_GRACE_FRAMES) {
        target = this._heldTarget;
      } else {
        target = 0.0;
      }
    }
    this.heat += (target - this.heat) * FireGestureState.HEAT_SMOOTHING;

    const flameTarget =
      trigger !== null ||
      (this._missFrames <= FireGestureState.L_GRACE_FRAMES && this.burning)
        ? 1.0
        : 0.0;
    this.flame += (flameTarget - this.flame) * FireGestureState.BURN_SMOOTHING;
    this.burning = this.flame > 0.02;
    if (!this.burning && trigger === null) this.triggerCenter = null;
    return [this.triggerCenter, this.heat, this.burning];
  }
}