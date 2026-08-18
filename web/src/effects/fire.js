import { randn } from "../gestures/utils.js";

const COLOR_STOPS = [
  // [age, r, g, b] (Python BGR converted to RGB), brightness 1.5 applied later
  [0.0, 255, 255, 255],
  [0.1, 250, 255, 255],
  [0.28, 150, 235, 255],
  [0.55, 55, 130, 235],
  [0.8, 25, 60, 160],
  [1.0, 10, 20, 70],
];

export class FireEffect {
  static MAX_PARTICLES = 800;
  static SPAWN_RATE = 300.0;
  static GLOW_SCALE = 4;
  static HALO_BLUR = 12;
  static BODY_BLUR = 4;
  static CORE_OPACITY = 0.9;
  static BODY_RADIUS_MULT = 1.7;
  static BUOYANCY = 70.0;
  static DRAG = 1.4;
  static TURBULENCE = 90.0;
  static SPEED_MIN = 120.0;
  static SPEED_MAX = 300.0;
  static SPREAD = 55.0;
  static LIFETIME_MIN = 0.8;
  static LIFETIME_MAX = 1.6;
  static RADIUS_MIN = 5.0;
  static RADIUS_MAX = 14.0;
  static EMITTER_RADIUS = 18.0;
  static BRIGHTNESS = 1.5;

  constructor() {
    this.posX = new Float32Array(FireEffect.MAX_PARTICLES);
    this.posY = new Float32Array(FireEffect.MAX_PARTICLES);
    this.velX = new Float32Array(FireEffect.MAX_PARTICLES);
    this.velY = new Float32Array(FireEffect.MAX_PARTICLES);
    this.lifetimes = new Float32Array(FireEffect.MAX_PARTICLES);
    this.maxLifetimes = new Float32Array(FireEffect.MAX_PARTICLES);
    this.radii = new Float32Array(FireEffect.MAX_PARTICLES);
    this.count = 0;
    this.burning = false;
  }

  ignite() {
    this.burning = true;
  }

  extinguish() {
    this.burning = false;
  }

  isBurning() {
    return this.burning;
  }

  _spawn(center, heat) {
    let n = Math.min(
      Math.floor((FireEffect.SPAWN_RATE * heat) / 30),
      FireEffect.MAX_PARTICLES - this.count
    );
    for (let i = 0; i < n; i++) {
      const speed =
        (FireEffect.SPEED_MIN +
          Math.random() * (FireEffect.SPEED_MAX - FireEffect.SPEED_MIN)) *
        (0.4 + 0.7 * heat);
      this.posX[this.count] =
        center[0] + (Math.random() * 2 - 1) * FireEffect.EMITTER_RADIUS;
      this.posY[this.count] =
        center[1] + (Math.random() * 2 - 1) * FireEffect.EMITTER_RADIUS;
      this.velX[this.count] = (Math.random() * 2 - 1) * FireEffect.SPREAD;
      this.velY[this.count] = -speed;
      this.lifetimes[this.count] =
        FireEffect.LIFETIME_MIN +
        Math.random() * (FireEffect.LIFETIME_MAX - FireEffect.LIFETIME_MIN);
      this.maxLifetimes[this.count] = this.lifetimes[this.count];
      this.radii[this.count] =
        (FireEffect.RADIUS_MIN +
          Math.random() * (FireEffect.RADIUS_MAX - FireEffect.RADIUS_MIN)) *
        (0.6 + 0.8 * heat);
      this.count += 1;
    }
  }

  _update(dt) {
    const drag = Math.exp(-FireEffect.DRAG * dt);
    const turb = FireEffect.TURBULENCE * dt;
    let alive = 0;
    for (let i = 0; i < this.count; i++) {
      this.lifetimes[i] -= dt;
      if (this.lifetimes[i] <= 0) continue;
      this.velY[i] += FireEffect.BUOYANCY * dt;
      this.velX[i] = this.velX[i] * drag + randn() * turb;
      const c = alive++;
      if (c !== i) {
        this.posX[c] = this.posX[i];
        this.posY[c] = this.posY[i];
        this.velX[c] = this.velX[i];
        this.velY[c] = this.velY[i];
        this.lifetimes[c] = this.lifetimes[i];
        this.maxLifetimes[c] = this.maxLifetimes[i];
        this.radii[c] = this.radii[i];
      }
    }
    this.count = alive;
    for (let i = 0; i < this.count; i++) {
      this.posX[i] += this.velX[i] * dt;
      this.posY[i] += this.velY[i] * dt;
    }
  }

  _colorRamp(age) {
    let lo = COLOR_STOPS[0];
    let hi = COLOR_STOPS[COLOR_STOPS.length - 1];
    for (let i = 0; i < COLOR_STOPS.length - 1; i++) {
      if (age >= COLOR_STOPS[i][0] && age <= COLOR_STOPS[i + 1][0]) {
        lo = COLOR_STOPS[i];
        hi = COLOR_STOPS[i + 1];
        break;
      }
    }
    const f =
      hi[0] - lo[0] > 1e-9 ? (age - lo[0]) / (hi[0] - lo[0]) : 0;
    const r = Math.round((lo[1] + (hi[1] - lo[1]) * f) * FireEffect.BRIGHTNESS);
    const g = Math.round((lo[2] + (hi[2] - lo[2]) * f) * FireEffect.BRIGHTNESS);
    const b = Math.round((lo[3] + (hi[3] - lo[3]) * f) * FireEffect.BRIGHTNESS);
    return `rgb(${Math.min(255, r)},${Math.min(255, g)},${Math.min(255, b)})`;
  }

  draw(ctx, W, H, center, heat = 1.0, dt = 1 / 30) {
    if (this.burning) this._spawn(center, heat);
    this._update(dt);
    if (this.count === 0) return;

    const s = FireEffect.GLOW_SCALE;
    const gw = Math.max(1, W / s);
    const gh = Math.max(1, H / s);
    const bodyCv = document.createElement("canvas");
    bodyCv.width = gw;
    bodyCv.height = gh;
    const bodyCtx = bodyCv.getContext("2d", { willReadFrequently: true });
    const haloCv = document.createElement("canvas");
    haloCv.width = gw;
    haloCv.height = gh;
    const haloCtx = haloCv.getContext("2d");

    bodyCtx.filter = `blur(${FireEffect.BODY_BLUR}px)`;
    haloCtx.filter = `blur(${FireEffect.HALO_BLUR}px)`;
    for (let i = 0; i < this.count; i++) {
      const age = 1 - this.lifetimes[i] / Math.max(this.maxLifetimes[i], 1e-6);
      const color = this._colorRamp(age);
      const px = this.posX[i] / s;
      const py = this.posY[i] / s;
      const bodyR = Math.max(1, (this.radii[i] * FireEffect.BODY_RADIUS_MULT) / s);
      const haloR = Math.max(1, this.radii[i] / s);
      bodyCtx.fillStyle = color;
      bodyCtx.beginPath();
      bodyCtx.arc(px, py, bodyR, 0, Math.PI * 2);
      bodyCtx.fill();
      haloCtx.fillStyle = color;
      haloCtx.beginPath();
      haloCtx.arc(px, py, haloR, 0, Math.PI * 2);
      haloCtx.fill();
    }

    // body alpha = brightness^1.2 * CORE_OPACITY (brightness of the color)
    const bodyImg = bodyCtx.getImageData(0, 0, gw, gh);
    const maskData = bodyCtx.createImageData(gw, gh);
    const src = bodyImg.data;
    const dst = maskData.data;
    for (let i = 0; i < src.length; i += 4) {
      const m = Math.max(src[i], src[i + 1], src[i + 2]);
      const a = Math.min(1, Math.pow(m / 255, 1.2) * FireEffect.CORE_OPACITY);
      dst[i] = dst[i + 1] = dst[i + 2] = 255;
      dst[i + 3] = Math.round(a * 255);
    }
    const maskCv = document.createElement("canvas");
    maskCv.width = gw;
    maskCv.height = gh;
    maskCv.getContext("2d").putImageData(maskData, 0, 0);
    bodyCtx.filter = "none";
    bodyCtx.globalCompositeOperation = "destination-in";
    bodyCtx.drawImage(maskCv, 0, 0);
    bodyCtx.globalCompositeOperation = "source-over";

    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(bodyCv, 0, 0, W, H);
    ctx.globalCompositeOperation = "lighter";
    ctx.drawImage(haloCv, 0, 0, W, H);
    ctx.restore();
  }
}