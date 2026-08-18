import { smoothstep } from "../gestures/utils.js";

export class Kaleidoscope {
  static SEGMENTS = 8;
  static ROTATION_SPEED = 0.5;
  static FADE_SPEED = 4.0;
  static RAMP_START = 0.55;
  static RAMP_WIDTH = 0.35;
  static MASK_FULL = 1.8;
  static MARGIN = 1.1;

  constructor() {
    this.phase = 0;
    this.blend = 0;
    this._tileCv = document.createElement("canvas");
    this._layerCv = document.createElement("canvas");
    this._maskCv = document.createElement("canvas");
  }

  apply(ctx, video, W, H, dt = 1 / 30, box = null, center = null, active = null) {
    if (active !== null) {
      const target = active ? 1 : 0;
      this.blend += (target - this.blend) * Math.min(1, dt * Kaleidoscope.FADE_SPEED);
    }
    if (this.blend < 0.01) return;

    let cx, cy;
    let tile = null;
    let keep = 0;
    let full = 0;
    if (box !== null) {
      const [bx0, by0, bx1, by1] = box;
      const bw = bx1 - bx0;
      const bh = by1 - by0;
      if (bw >= 4 && bh >= 4) {
        cx = (bx0 + bx1) / 2;
        cy = (by0 + by1) / 2;
        const hw = (bw / 2) * Kaleidoscope.MARGIN;
        const hh = (bh / 2) * Kaleidoscope.MARGIN;
        const x0 = Math.max(0, Math.round(cx - hw));
        const x1 = Math.min(W, Math.round(cx + hw));
        const y0 = Math.max(0, Math.round(cy - hh));
        const y1 = Math.min(H, Math.round(cy + hh));
        if (x1 - x0 >= 4 && y1 - y0 >= 4) {
          const tw = x1 - x0;
          const th = y1 - y0;
          keep = Math.max(tw, th) / 2;
          full = keep * Kaleidoscope.MASK_FULL;
          tile = this._buildMirrorTile(video, x0, y0, tw, th);
        }
      }
    }
    if (tile === null) {
      cx = center !== null ? center[0] : W / 2;
      cy = center !== null ? center[1] : H / 2;
      keep = 0;
      full = Math.sqrt(Math.max(cx, W - cx) ** 2 + Math.max(cy, H - cy) ** 2);
    }

    this.phase += dt * Kaleidoscope.ROTATION_SPEED;

    // kaleido layer: rotated pattern
    const layerCv = this._layerCv;
    layerCv.width = W;
    layerCv.height = H;
    const lctx = layerCv.getContext("2d");
    if (tile !== null) {
      // mirror-tile: 2x2 mirrored block repeated, rotated around the person
      const pat = lctx.createPattern(tile, "repeat");
      lctx.save();
      lctx.translate(cx, cy);
      lctx.rotate(-this.phase);
      lctx.translate(-cx, -cy);
      lctx.fillStyle = pat;
      lctx.fillRect(0, 0, W, H);
      lctx.restore();
    } else {
      // frame mode: radial angle-fold into SEGMENTS wedges
      const sector = (Math.PI * 2) / Kaleidoscope.SEGMENTS;
      lctx.save();
      lctx.translate(cx, cy);
      lctx.rotate(-this.phase);
      for (let i = 0; i < Kaleidoscope.SEGMENTS; i++) {
        const a0 = i * sector;
        const a1 = (i + 1) * sector;
        lctx.save();
        lctx.beginPath();
        lctx.moveTo(0, 0);
        lctx.arc(0, 0, full, a0, a1);
        lctx.closePath();
        lctx.clip();
        if (i % 2 === 1) {
          const mid = (a0 + a1) / 2;
          lctx.translate(Math.cos(mid) * full, Math.sin(mid) * full);
          lctx.rotate(mid);
          lctx.scale(-1, 1);
          lctx.rotate(-mid);
          lctx.translate(-Math.cos(mid) * full, -Math.sin(mid) * full);
        }
        lctx.drawImage(video, -full, -full, full * 2, full * 2);
        lctx.restore();
      }
      lctx.restore();
    }

    // radial alpha mask
    const maskCv = this._maskCv;
    maskCv.width = W;
    maskCv.height = H;
    const mctx = maskCv.getContext("2d");
    if (tile !== null && full > keep) {
      const g = mctx.createRadialGradient(cx, cy, keep, cx, cy, full);
      g.addColorStop(0, "rgba(255,255,255,0)");
      g.addColorStop(1, "rgba(255,255,255,1)");
      mctx.fillStyle = g;
      mctx.fillRect(0, 0, W, H);
    } else {
      // frame mode: center unchanged, fold ramps in toward the rim
      const r0 = Kaleidoscope.RAMP_START;
      const r1 = Kaleidoscope.RAMP_START + Kaleidoscope.RAMP_WIDTH;
      const g = mctx.createRadialGradient(cx, cy, 0, cx, cy, full);
      g.addColorStop(Math.max(0, r0), "rgba(255,255,255,0)");
      g.addColorStop(Math.min(1, r1), "rgba(255,255,255,1)");
      mctx.fillStyle = g;
      mctx.fillRect(0, 0, W, H);
    }

    // mask the layer, then blend with globalAlpha = blend
    mctx.globalCompositeOperation = "source-in";
    mctx.drawImage(layerCv, 0, 0);
    ctx.save();
    ctx.globalAlpha = this.blend;
    ctx.drawImage(maskCv, 0, 0);
    ctx.restore();
  }

  _buildMirrorTile(video, x0, y0, tw, th) {
    const cv = this._tileCv;
    cv.width = tw * 2;
    cv.height = th * 2;
    const tc = cv.getContext("2d");
    tc.clearRect(0, 0, tw * 2, th * 2);

    // top-left: original crop
    tc.drawImage(video, x0, y0, tw, th, 0, 0, tw, th);

    // top-right: horizontally mirrored
    tc.save();
    tc.scale(-1, 1);
    tc.drawImage(video, x0, y0, tw, th, -2 * tw, 0, tw, th);
    tc.restore();

    // bottom-left: vertically mirrored
    tc.save();
    tc.scale(1, -1);
    tc.drawImage(video, x0, y0, tw, th, 0, -2 * th, tw, th);
    tc.restore();

    // bottom-right: both mirrored
    tc.save();
    tc.scale(-1, -1);
    tc.drawImage(video, x0, y0, tw, th, -2 * tw, -2 * th, tw, th);
    tc.restore();

    return cv;
  }
}