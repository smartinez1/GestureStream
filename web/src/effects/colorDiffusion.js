export class ColorDiffusion {
  static DECAY = 0.9;
  static MASK_DECAY = 0.55;
  static OUTER_PAD = 70;
  static INNER_PAD = 16;
  static HUE_SPEED = 55.0;
  static INTENSITY = 0.4;
  static HUE_BASE = 180.0;
  static HUE_RANGE = 120.0;

  constructor() {
    this.trail = null;
    this._rimCv = document.createElement("canvas");
    this._blurCv = document.createElement("canvas");
  }

  _ensure(hw, hh) {
    if (this.trail === null || this.trail.length !== hw * hh * 3) {
      this.trail = new Float32Array(hw * hh * 3);
    }
  }

  clear() {
    if (this.trail !== null) this.trail.fill(0);
  }

  isActive() {
    if (this.trail === null) return false;
    for (let i = 0; i < this.trail.length; i += 3) {
      if (this.trail[i] > 1 || this.trail[i + 1] > 1 || this.trail[i + 2] > 1) {
        return true;
      }
    }
    return false;
  }

  _paint(maskCv, t, hw, hh) {
    this._rimCv.width = hw;
    this._rimCv.height = hh;
    const rctx = this._rimCv.getContext("2d", { willReadFrequently: true });
    // distance-transform approximation: blurred mask alpha = rim falloff
    rctx.filter = "blur(35px)";
    rctx.drawImage(maskCv, 0, 0, hw, hh);
    rctx.filter = "none";
    const rim = rctx.getImageData(0, 0, hw, hh).data;

    const mimg = maskCv.getContext("2d").getImageData(0, 0, hw, hh).data;
    let sumX = 0;
    let sumY = 0;
    let n = 0;
    for (let y = 0; y < hh; y++) {
      for (let x = 0; x < hw; x++) {
        if (mimg[(y * hw + x) * 4] >= 128) {
          sumX += x;
          sumY += y;
          n += 1;
        }
      }
    }
    if (n === 0) return;
    const cx = sumX / n;
    const cy = sumY / n;

    const trail = this.trail;
    for (let y = 0; y < hh; y++) {
      for (let x = 0; x < hw; x++) {
        const i = (y * hw + x) * 4;
        const a = rim[i + 3] / 255;
        if (a <= 0) continue;
        const ang = Math.atan2(y - cy, x - cx);
        const hue =
          ColorDiffusion.HUE_BASE +
          ((ang * 57.2958 + t * ColorDiffusion.HUE_SPEED) %
            ColorDiffusion.HUE_RANGE);
        const [r, g, b] = hsv2rgb(hue, 1, 1);
        const wgt = a * a * ColorDiffusion.INTENSITY;
        const j = (y * hw + x) * 3;
        trail[j] += r * wgt;
        trail[j + 1] += g * wgt;
        trail[j + 2] += b * wgt;
      }
    }
  }

  apply(ctx, maskCv, t, dt = 1 / 30, enabled = true, W, H) {
    const hw = W >> 1;
    const hh = H >> 1;
    this._ensure(hw, hh);
    const trail = this.trail;
    for (let i = 0; i < trail.length; i++) trail[i] *= ColorDiffusion.DECAY;

    if (maskCv !== null) {
      const mimg = maskCv.getContext("2d").getImageData(0, 0, hw, hh).data;
      for (let y = 0; y < hh; y++) {
        for (let x = 0; x < hw; x++) {
          if (mimg[(y * hw + x) * 4] >= 128) {
            const j = (y * hw + x) * 3;
            trail[j] *= ColorDiffusion.MASK_DECAY;
            trail[j + 1] *= ColorDiffusion.MASK_DECAY;
            trail[j + 2] *= ColorDiffusion.MASK_DECAY;
          }
        }
      }
      if (enabled) this._paint(maskCv, t, hw, hh);
    }

    const img = new ImageData(hw, hh);
    for (let i = 0; i < hw * hh; i++) {
      const j = i * 3;
      img.data[i * 4] = Math.min(255, trail[j]);
      img.data[i * 4 + 1] = Math.min(255, trail[j + 1]);
      img.data[i * 4 + 2] = Math.min(255, trail[j + 2]);
      img.data[i * 4 + 3] = 255;
    }
    this._blurCv.width = hw;
    this._blurCv.height = hh;
    const bctx = this._blurCv.getContext("2d");
    bctx.putImageData(img, 0, 0);
    bctx.filter = "blur(5px)";
    bctx.drawImage(this._blurCv, 0, 0);
    bctx.filter = "none";

    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(this._blurCv, 0, 0, W, H);
    ctx.restore();
  }
}

function hsv2rgb(h, s, v) {
  h = ((h % 360) + 360) % 360;
  const c = v * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = v - c;
  let r, g, b;
  if (h < 60) [r, g, b] = [c, x, 0];
  else if (h < 120) [r, g, b] = [x, c, 0];
  else if (h < 180) [r, g, b] = [0, c, x];
  else if (h < 240) [r, g, b] = [0, x, c];
  else if (h < 300) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  return [(r + m) * 255, (g + m) * 255, (b + m) * 255];
}