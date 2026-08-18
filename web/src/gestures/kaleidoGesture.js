import { palmCenter, pointsUp, dist2d } from "./utils.js";

export class KaleidoscopeGestureState {
  static POINTS_UP = 1.0;
  static PALM_DX = 0.12;
  static PALM_DY = 0.18;
  static THUMB_DIST = 0.12;
  static HOLD_TIME = 0.6;

  constructor() {
    this.enabled = false;
    this.praying = false;
    this.poseStart = null;
    this.toggleFired = false;
  }

  detectPrayingPose(hands) {
    if (hands.length < 2) return false;
    const [a, b] = hands;
    if (!(pointsUp(a) && pointsUp(b))) return false;
    const pa = palmCenter(a);
    const pb = palmCenter(b);
    if (
      Math.abs(pa.x - pb.x) > KaleidoscopeGestureState.PALM_DX ||
      Math.abs(pa.y - pb.y) > KaleidoscopeGestureState.PALM_DY
    ) {
      return false;
    }
    return dist2d(a[4], b[4]) < KaleidoscopeGestureState.THUMB_DIST;
  }

  update(hands, now) {
    let fired = false;
    const praying = this.detectPrayingPose(hands);
    if (praying) {
      if (this.poseStart === null) {
        this.poseStart = now;
        this.toggleFired = false;
      } else if (!this.toggleFired && now - this.poseStart >= KaleidoscopeGestureState.HOLD_TIME) {
        this.enabled = !this.enabled;
        this.toggleFired = true;
        fired = true;
      }
    } else {
      this.poseStart = null;
      this.toggleFired = false;
    }
    this.praying = praying;
    return fired;
  }
}