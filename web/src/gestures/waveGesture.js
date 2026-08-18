import { palmCenter, dist2d } from "./utils.js";

export class WaveGestureState {
  static WAVE_AMPLITUDE = 0.09;
  static WAVE_SWINGS = 3;
  static WAVE_WINDOW = 2.5;
  static WAVE_TIMEOUT = 1.2;
  static WAVE_VEL_EPS = 0.02;
  static WAVE_GRACE = 0.35;
  static WAVE_DEAD = 0.005;

  constructor() {
    this.waveEnabled = false;
    this.waving = false;
    this.waveSamples = new CircularBuffer(90);
    this.axisSwings = { x: 0, y: 0, d: 0 };
    this.waveStartT = { x: null, y: null, d: null };
    this.waveLastT = { x: null, y: null, d: null };
    this.waveDropT = null;
    this.waveTrack = {
      x: { peak: null, trough: null, dir: null, prev: null },
      y: { peak: null, trough: null, dir: null, prev: null },
      d: { peak: null, trough: null, dir: null, prev: null },
    };
  }

  _waveReset() {
    this.waveSamples.clear();
    this.axisSwings = { x: 0, y: 0, d: 0 };
    this.waveStartT = { x: null, y: null, d: null };
    this.waveLastT = { x: null, y: null, d: null };
    this.waveDropT = null;
    for (const s of Object.values(this.waveTrack)) {
      s.peak = s.trough = s.prev = null;
      s.dir = null;
    }
  }

  _trackAxis(key, v, now) {
    const s = this.waveTrack[key];
    if (s.peak === null) {
      s.peak = s.trough = s.prev = v;
      s.dir = null;
      return;
    }
    if (v > s.peak) s.peak = v;
    if (v < s.trough) s.trough = v;
    const delta = v - s.prev;
    s.prev = v;
    if (Math.abs(delta) < WaveGestureState.WAVE_DEAD) return;
    const nd = delta > 0 ? "up" : "down";
    if (s.dir !== null && nd !== s.dir) {
      if (s.peak - s.trough >= WaveGestureState.WAVE_AMPLITUDE) {
        this.axisSwings[key] += 1;
        if (this.axisSwings[key] === 1) this.waveStartT[key] = now;
        this.waveLastT[key] = now;
        s.peak = s.trough = v;
      }
    }
    s.dir = nd;
  }

  update(hands, now) {
    if (hands.length < 2) {
      this.waving = false;
      if (this.waveSamples.length > 0) {
        if (this.waveDropT === null) this.waveDropT = now;
        else if (now - this.waveDropT > WaveGestureState.WAVE_GRACE) this._waveReset();
      }
      return false;
    }
    this.waveDropT = null;

    const palmA = palmCenter(hands[0]);
    const palmB = palmCenter(hands[1]);
    const avg = [(palmA.x + palmB.x) / 2, (palmA.y + palmB.y) / 2];
    const sep = dist2d(palmA, palmB);
    this.waveSamples.push([now, [avg[0], avg[1], sep]]);

    let moving = false;
    if (this.waveSamples.length >= 2) {
      const [t0, p0] = this.waveSamples.at(-2);
      const [t1, p1] = this.waveSamples.at(-1);
      const dt = t1 - t0;
      if (dt > 0) {
        const d = Math.hypot(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]);
        moving = d / dt >= WaveGestureState.WAVE_VEL_EPS;
      }
    }

    this._trackAxis("x", avg[0], now);
    this._trackAxis("y", avg[1], now);
    this._trackAxis("d", sep, now);
    this.waving =
      moving || this.axisSwings.x > 0 || this.axisSwings.y > 0 || this.axisSwings.d > 0;

    for (const key of ["x", "y", "d"]) {
      if (
        this.axisSwings[key] > 0 &&
        now - this.waveLastT[key] > WaveGestureState.WAVE_TIMEOUT
      ) {
        this.axisSwings[key] = 0;
        this.waveStartT[key] = null;
        this.waveLastT[key] = null;
      }
      if (
        this.axisSwings[key] >= WaveGestureState.WAVE_SWINGS &&
        now - this.waveStartT[key] <= WaveGestureState.WAVE_WINDOW
      ) {
        this.waveEnabled = !this.waveEnabled;
        this._waveReset();
        return true;
      }
    }
    return false;
  }
}

class CircularBuffer {
  constructor(maxlen) {
    this.maxlen = maxlen;
    this.buf = new Array(maxlen);
    this.start = 0;
    this.length = 0;
  }
  push(v) {
    if (this.length === this.maxlen) {
      this.buf[this.start] = v;
      this.start = (this.start + 1) % this.maxlen;
    } else {
      this.buf[(this.start + this.length) % this.maxlen] = v;
      this.length += 1;
    }
  }
  at(i) {
    const n = i < 0 ? this.length + i : i;
    return this.buf[(this.start + n) % this.maxlen];
  }
  clear() {
    this.start = 0;
    this.length = 0;
  }
}

export class PersonSegmenter {
  static MIN_AREA = 0.002;
  static CONFIDENCE_THRESHOLD = 0.7;
  static MARGIN = 1.1;

  constructor(segmenter) {
    this.segmenter = segmenter;
    this.mask = null; // binary mask canvas at (w/2, h/2)
    this._maskCv = document.createElement("canvas");
    this._tmpCv = document.createElement("canvas");
  }

  bbox(video, w, h, ts) {
    const mw = w >> 1;
    const mh = h >> 1;
    const result = this.segmenter.segmentForVideo(video, ts);
    const conf = result.confidenceMasks?.[0];
    if (conf === undefined) return null;
    const sw = conf.width;
    const sh = conf.height;
    const data = conf.getAsFloat32Array();
    const thr = PersonSegmenter.CONFIDENCE_THRESHOLD;

    this._maskCv.width = sw;
    this._maskCv.height = sh;
    let mctx = this._maskCv.getContext("2d", { willReadFrequently: true });
    const img = mctx.createImageData(sw, sh);
    for (let i = 0; i < data.length; i++) {
      const v = data[i] >= thr ? 255 : 0;
      img.data[i * 4] = v;
      img.data[i * 4 + 1] = v;
      img.data[i * 4 + 2] = v;
      img.data[i * 4 + 3] = 255;
    }
    mctx.putImageData(img, 0, 0);

    // morphological close approximation: blur + re-threshold fills small gaps
    this._tmpCv.width = mw;
    this._tmpCv.height = mh;
    const tctx = this._tmpCv.getContext("2d", { willReadFrequently: true });
    tctx.imageSmoothingEnabled = true;
    tctx.filter = "blur(2px)";
    tctx.drawImage(this._maskCv, 0, 0, mw, mh);
    tctx.filter = "none";
    const small = tctx.getImageData(0, 0, mw, mh);
    let minX = mw;
    let minY = mh;
    let maxX = -1;
    let maxY = -1;
    for (let y = 0; y < mh; y++) {
      for (let x = 0; x < mw; x++) {
        if (small.data[(y * mw + x) * 4] >= 128) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    this.mask = this._tmpCv; // binary mask at half res
    if (maxX < 0) return null;
    const bw = (maxX - minX + 1) * 2;
    const bh = (maxY - minY + 1) * 2;
    if (bw * bh < PersonSegmenter.MIN_AREA * w * h) return null;
    return [minX * 2, minY * 2, (maxX + 1) * 2, (maxY + 1) * 2];
  }
}

export class PersonTrackerState {
  static SEGMENT_EVERY = 3;
  static MAX_MISSED = 2;
  static BOX_SMOOTH_WINDOW = 5;
  static ANCHOR_SMOOTHING = 0.15;

  constructor(segmenter) {
    this.segmenter = segmenter;
    this.frameCount = 0;
    this.missed = 0;
    this.boxHistory = new CircularBuffer(PersonTrackerState.BOX_SMOOTH_WINDOW);
    this.personRect = null;
    this.center = null;
  }

  update(video, w, h, ts) {
    this.frameCount += 1;
    if (this.frameCount % PersonTrackerState.SEGMENT_EVERY !== 0) return;
    const box = this.segmenter.bbox(video, w, h, ts);
    if (box === null) {
      this.missed += 1;
      if (this.missed > PersonTrackerState.MAX_MISSED) {
        this.personRect = null;
        this.center = null;
        this.boxHistory.clear();
      }
      return;
    }
    this.missed = 0;
    this.boxHistory.push(box);
    const arr = [];
    for (let i = 0; i < 4; i++) {
      const col = [];
      for (let j = 0; j < this.boxHistory.length; j++) col.push(this.boxHistory.at(j)[i]);
      col.sort((a, b) => a - b);
      arr.push(col[col.length >> 1]);
    }
    const target = arr;
    if (this.personRect === null) {
      this.personRect = [...target];
    } else {
      const a = PersonTrackerState.ANCHOR_SMOOTHING;
      for (let i = 0; i < 4; i++) {
        this.personRect[i] += (target[i] - this.personRect[i]) * a;
      }
    }
    const [x0, y0, x1, y1] = this.personRect;
    this.center = [(x0 + x1) / 2, (y0 + y1) / 2];
  }
}