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
    this._blurCv = document.createElement("canvas");
    this._blurOutCv = document.createElement("canvas");
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

  static _edt(maskR, w, h, seedIsPerson) {
    const INF = 9999;
    const dist = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) {
      const isPerson = maskR[i * 4] >= 128;
      dist[i] = isPerson === seedIsPerson ? 0 : INF;
    }
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (dist[i] === 0) continue;
        if (x > 0) dist[i] = Math.min(dist[i], dist[i - 1] + 1);
        if (y > 0) dist[i] = Math.min(dist[i], dist[i - w] + 1);
        if (x > 0 && y > 0) dist[i] = Math.min(dist[i], dist[i - w - 1] + 1);
        if (x < w - 1 && y > 0) dist[i] = Math.min(dist[i], dist[i - w + 1] + 1);
      }
    }
    for (let y = h - 1; y >= 0; y--) {
      for (let x = w - 1; x >= 0; x--) {
        const i = y * w + x;
        if (x < w - 1) dist[i] = Math.min(dist[i], dist[i + 1] + 1);
        if (y < h - 1) dist[i] = Math.min(dist[i], dist[i + w] + 1);
        if (x < w - 1 && y < h - 1) dist[i] = Math.min(dist[i], dist[i + w + 1] + 1);
        if (x > 0 && y < h - 1) dist[i] = Math.min(dist[i], dist[i + w - 1] + 1);
      }
    }
    return dist;
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

    // outerDt: distance of each pixel to the nearest PERSON pixel → the
    // OUTER rim (bg pixels: how far past the silhouette edge). Seeds person.
    // innerDt: distance of each pixel to the nearest BG pixel → the INNER
    // rim (person pixels: how far inside the silhouette). Seeds bg.
    const outerDt = ColorDiffusion._edt(mdata, hw, hh, true);
    const innerDt = ColorDiffusion._edt(mdata, hw, hh, false);
    const outerThresh = ColorDiffusion.OUTER_PAD / 2;
    const innerThresh = ColorDiffusion.INNER_PAD / 2;

    const trail = this.trail;
    const INTENSITY = ColorDiffusion.INTENSITY;
    for (let y = 0; y < hh; y++) {
      for (let x = 0; x < hw; x++) {
        const idx = y * hw + x;
        const isPerson = mdata[idx * 4] >= 128;
        const dt = isPerson ? innerDt[idx] : outerDt[idx];
        const thresh = isPerson ? innerThresh : outerThresh;
        if (dt >= thresh) continue;
        let s = 1.0 - dt / thresh;
        s = s * s;
        const wgt = s * INTENSITY;
        const ang = Math.atan2(y - cy, x - cx);
        const hue =
          ColorDiffusion.HUE_BASE +
          ((ang * 57.2958 + t * ColorDiffusion.HUE_SPEED) %
            ColorDiffusion.HUE_RANGE);
        const [r, g, b] = hsv2rgb(((hue % 360) + 360) % 360, 1, 1);
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

    this._blurOutCv.width = hw;
    this._blurOutCv.height = hh;
    const bctx2 = this._blurOutCv.getContext("2d");
    bctx2.filter = "blur(3px)";
    bctx2.drawImage(this._blurCv, 0, 0);
    bctx2.filter = "none";

    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(this._blurOutCv, 0, 0, W, H);
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