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
    this._maxVal = 0;
    this._rimCv = document.createElement("canvas");
    this._blurCv = document.createElement("canvas");
    this._outerSrcCv = document.createElement("canvas");
    this._innerSrcCv = document.createElement("canvas");
    this._innerRimCv = document.createElement("canvas");
  }

  _ensure(hw, hh) {
    if (this.trail === null || this.trail.length !== hw * hh * 3) {
      this.trail = new Float32Array(hw * hh * 3);
    }
  }

  clear() {
    if (this.trail !== null) this.trail.fill(0);
    this._maxVal = 0;
  }

  isActive() {
    return this._maxVal > 1;
  }

  _paint(maskCv, t, hw, hh) {
    const mdata = maskCv
      .getContext("2d", { willReadFrequently: true })
      .getImageData(0, 0, hw, hh).data;

    let sx = 0;
    let sy = 0;
    let n = 0;
    for (let i = 0; i < hw * hh; i++) {
      if (mdata[i * 4] >= 128) {
        sx += i % hw;
        sy += Math.floor(i / hw);
        n += 1;
      }
    }
    if (n === 0) return;
    const cx = sx / n;
    const cy = sy / n;

    // --- outer rim: person=opaque, bg=transparent → blur outward ---
    const outerImg = new ImageData(hw, hh);
    for (let i = 0; i < hw * hh; i++) {
      outerImg.data[i * 4] = outerImg.data[i * 4 + 1] = outerImg.data[i * 4 + 2] = 255;
      outerImg.data[i * 4 + 3] = mdata[i * 4] >= 128 ? 255 : 0;
    }
    this._outerSrcCv.width = hw;
    this._outerSrcCv.height = hh;
    this._outerSrcCv.getContext("2d").putImageData(outerImg, 0, 0);
    this._rimCv.width = hw;
    this._rimCv.height = hh;
    const rctx = this._rimCv.getContext("2d", { willReadFrequently: true });
    rctx.clearRect(0, 0, hw, hh);
    rctx.filter = `blur(${Math.ceil(ColorDiffusion.OUTER_PAD / 6)}px)`;
    rctx.drawImage(this._outerSrcCv, 0, 0);
    rctx.filter = "none";
    const outerRim = rctx.getImageData(0, 0, hw, hh).data;

    // --- inner rim: bg=opaque, person=transparent → blur inward ---
    const innerImg = new ImageData(hw, hh);
    for (let i = 0; i < hw * hh; i++) {
      innerImg.data[i * 4] = innerImg.data[i * 4 + 1] = innerImg.data[i * 4 + 2] = 255;
      innerImg.data[i * 4 + 3] = mdata[i * 4] >= 128 ? 0 : 255;
    }
    this._innerSrcCv.width = hw;
    this._innerSrcCv.height = hh;
    this._innerSrcCv.getContext("2d").putImageData(innerImg, 0, 0);
    this._innerRimCv.width = hw;
    this._innerRimCv.height = hh;
    const ictx = this._innerRimCv.getContext("2d", { willReadFrequently: true });
    ictx.clearRect(0, 0, hw, hh);
    ictx.filter = `blur(${Math.ceil(ColorDiffusion.INNER_PAD / 6)}px)`;
    ictx.drawImage(this._innerSrcCv, 0, 0);
    ictx.filter = "none";
    const innerRim = ictx.getImageData(0, 0, hw, hh).data;

    // --- paint trail ---
    const trail = this.trail;
    const INTENSITY = ColorDiffusion.INTENSITY;
    for (let y = 0; y < hh; y++) {
      for (let x = 0; x < hw; x++) {
        const idx = y * hw + x;
        const isPerson = mdata[idx * 4] >= 128;
        // blur of a step edge peaks at ~127 alpha; normalize so the
        // silhouette edge reaches 1.0 like the Python distance transform
        const a = Math.min(
          1,
          (isPerson ? innerRim[idx * 4 + 3] : outerRim[idx * 4 + 3]) / 127.5
        );
        if (a < 0.005) continue;
        const wgt = a * a * INTENSITY;
        const ang = Math.atan2(y - cy, x - cx);
        const hue =
          ColorDiffusion.HUE_BASE +
          ((ang * 57.2958 + t * ColorDiffusion.HUE_SPEED) %
            ColorDiffusion.HUE_RANGE);
        const [r, g, b] = hsv2rgb(hue, 1, 1);
        trail[idx * 3] += r * wgt;
        trail[idx * 3 + 1] += g * wgt;
        trail[idx * 3 + 2] += b * wgt;
      }
    }
  }

  apply(ctx, maskCv, t, dt = 1 / 30, enabled = true, W, H) {
    const hw = W >> 1;
    const hh = H >> 1;
    this._ensure(hw, hh);
    const trail = this.trail;
    for (let i = 0; i < trail.length; i++) trail[i] *= ColorDiffusion.DECAY;
    let newMax = 0;
    for (let i = 0; i < trail.length; i++) {
      if (trail[i] > newMax) newMax = trail[i];
    }
    this._maxVal = newMax;

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