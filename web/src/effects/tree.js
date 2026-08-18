import { clip, smoothstep } from "../gestures/utils.js";

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randnFrom(rng) {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function vec3(a, b, c) {
  return [a, b, c];
}

function vAdd(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function vSub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function vScale(a, s) {
  return [a[0] * s, a[1] * s, a[2] * s];
}

function vLerp(a, b, t) {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

function vLen(a) {
  return Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

function vNorm(a) {
  const len = vLen(a);
  if (len < 1e-12) return [0, 0, 0];
  return vScale(a, 1 / len);
}

// Python BGR palettes converted to RGB
const BARK_BOTTOM = [90, 60, 35];
const BARK_TOP = [150, 105, 60];
const BARK_L1 = [120, 85, 50];
const BARK_L2 = [140, 100, 58];
const LEAF_GREENS = [
  [65, 130, 50],
  [80, 160, 70],
  [100, 190, 95],
  [110, 205, 120],
  [120, 215, 150],
  [135, 225, 175],
];
const LEAF_ACCENTS = [
  [190, 120, 0],
  [55, 135, 165],
  [90, 165, 200],
];

export class VoxelTree {
  static TRUNK_HEIGHT = 180.0;
  static TRUNK_RADIUS = 22.0;
  static SEG_LEN = 15.0;
  static BRANCHES_L1 = 5;
  static BRANCHES_L2 = 3;
  static L2_LENGTH = 0.55;
  static LEAF_TWIG = 7;
  static LEAF_CROWN = 90;
  static GROW_DURATION = 3.0;
  static FADE_DURATION = 2.0;
  static MAX_HEIGHT_SCALE = 1.6;
  static FALL_DIST = 60.0;
  static SWAY = 0.7;
  static SWAY_WIND = 1.7;

  constructor(seed = 1337) {
    this.rng = mulberry32(seed);
    this.segments = [];
    this.leaves = [];
    this.active = false;
    this.growing = false;
    this.fading = false;
    this.growTime = 0.0;
    this.fadeTime = 0.0;
    this._trunkH = VoxelTree.TRUNK_HEIGHT;
  }

  spawn(heightScale = 1.0) {
    this.segments = [];
    this.leaves = [];
    this._build(heightScale);
    this._trunkH = VoxelTree.TRUNK_HEIGHT * heightScale;
    this.active = true;
    this.growing = true;
    this.fading = false;
    this.growTime = 0.0;
    this.fadeTime = 0.0;
  }

  startFade() {
    if (this.active && !this.fading) {
      this.fading = true;
      this.growing = false;
      this.fadeTime = 0.0;
    }
  }

  isActive() {
    return this.active;
  }

  _lerpColor(c1, c2, f) {
    return [
      Math.round(c1[0] + (c2[0] - c1[0]) * f),
      Math.round(c1[1] + (c2[1] - c1[1]) * f),
      Math.round(c1[2] + (c2[2] - c1[2]) * f),
    ];
  }

  _jitter(color, amount) {
    return [
      Math.round(clip(color[0] + (this.rng() * 2 - 1) * amount, 0, 255)),
      Math.round(clip(color[1] + (this.rng() * 2 - 1) * amount, 0, 255)),
      Math.round(clip(color[2] + (this.rng() * 2 - 1) * amount, 0, 255)),
    ];
  }

  _pointOnTrunk(f) {
    const trunk = this.segments.filter((s) => s.gen === 0);
    const n = trunk.length;
    if (n === 0) return [0, 0, 0];
    f = clip(f, 0, 0.999);
    const idx = Math.floor(f * n);
    const t = f * n - idx;
    const seg = trunk[idx];
    return vLerp(seg.a, seg.b, t);
  }

  _branch(origin, direction, length, radius, gen, revealBase, heightScale) {
    const rng = this.rng;
    const segLen = VoxelTree.SEG_LEN * heightScale;
    const nSeg = Math.max(3, Math.floor(length / segLen));
    let pos = [...origin];
    let d = [...direction];
    const bend = gen === 1 ? 0.06 : 0.1;
    for (let i = 0; i < nSeg; i++) {
      const f = i / nSeg;
      d[1] += bend;
      d[1] = Math.max(d[1], 0.12);
      d = vAdd(d, vScale(vec3(randnFrom(rng), randnFrom(rng), randnFrom(rng)), 0.05 * (gen + 1)));
      d = vNorm(d);
      const end = vAdd(pos, vScale(d, length / nSeg));
      const r = Math.max(radius * (1 - 0.65 * f), 0.8);
      const color = gen === 1 ? BARK_L1 : BARK_L2;
      this.segments.push({
        a: pos,
        b: end,
        r,
        gen,
        color,
        reveal: revealBase + f * 0.2,
        phase: rng() * Math.PI * 2,
      });
      pos = end;
    }
    return [pos, d];
  }

  _leaf(pos, radius, reveal) {
    const rng = this.rng;
    const hf = clip(pos[1] / VoxelTree.TRUNK_HEIGHT + 0.5, 0, 1);
    let color;
    if (rng() < 0.1) {
      color = LEAF_ACCENTS[Math.floor(rng() * LEAF_ACCENTS.length)];
    } else {
      const base = Math.floor(hf * (LEAF_GREENS.length - 1));
      color = this._jitter(LEAF_GREENS[base], 12);
    }
    this.leaves.push({
      pos: [...pos],
      r: radius,
      color,
      reveal,
      phase: rng() * Math.PI * 2,
    });
  }

  _build(heightScale) {
    const rng = this.rng;
    const trunkH = VoxelTree.TRUNK_HEIGHT * heightScale;
    const trunkR = VoxelTree.TRUNK_RADIUS * heightScale;

    let pos = vec3(0, -trunkH / 2, 0);
    const lean = rng() * 0.24 - 0.12;
    let d = vNorm(vec3(lean, 1, rng() * 0.16 - 0.08));
    const trunkLen = trunkH * 0.52;
    const nSeg = Math.max(4, Math.floor(trunkLen / (VoxelTree.SEG_LEN * heightScale)));
    for (let i = 0; i < nSeg; i++) {
      const f = i / nSeg;
      d[0] += (lean * 0.5 - d[0]) * 0.15;
      d[1] += 0.03;
      d = vAdd(d, vScale(vec3(randnFrom(rng), randnFrom(rng), randnFrom(rng)), 0.015));
      d = vNorm(d);
      const end = vAdd(pos, vScale(d, trunkLen / nSeg));
      let r = trunkR * (1 - 0.72 * f);
      if (i === 0) r = trunkR * 1.2;
      const color = this._jitter(this._lerpColor(BARK_BOTTOM, BARK_TOP, f), 6);
      this.segments.push({
        a: pos,
        b: end,
        r: Math.max(r, 1.2),
        gen: 0,
        color,
        reveal: f * 0.22,
        phase: rng() * Math.PI * 2,
      });
      pos = end;
    }

    for (let b = 0; b < VoxelTree.BRANCHES_L1; b++) {
      const f = 0.45 + rng() * 0.5;
      const origin = this._pointOnTrunk(f);
      const az = rng() * Math.PI * 2;
      let bd = vNorm(vec3(Math.cos(az), 0.5 + rng() * 0.35, Math.sin(az)));
      bd[1] = Math.abs(bd[1]);
      bd = vNorm(bd);
      const length = trunkH * (0.3 + rng() * 0.12);
      const r = trunkR * (0.2 + rng() * 0.08);
      const [end1, bdOut] = this._branch(origin, bd, length, r, 1, 0.24, heightScale);
      bd = bdOut;

      for (let i = 0; i < 6; i++) {
        const t = 0.5 + rng() * 0.5;
        this._leaf(vLerp(origin, end1, t), 2 + rng() * 1.2, 0.78 + rng() * 0.18);
      }

      for (let i = 0; i < VoxelTree.BRANCHES_L2; i++) {
        let sd = vAdd(bd, vScale(vec3(randnFrom(rng), randnFrom(rng), randnFrom(rng)), 0.25));
        sd[1] += 0.25;
        sd = vNorm(sd);
        const subLen = length * VoxelTree.L2_LENGTH * (0.8 + rng() * 0.4);
        const [end2] = this._branch(end1, sd, subLen, r * 0.55, 2, 0.52, heightScale);
        for (let j = 0; j < VoxelTree.LEAF_TWIG; j++) {
          let off = vec3(randnFrom(rng), randnFrom(rng), randnFrom(rng));
          const len = vLen(off) + 1e-9;
          off = vScale(off, (1 / len) * rng() * trunkR * 1.1 * heightScale);
          this._leaf(vAdd(end2, off), 2.2 + rng() * 1.4, 0.86 + rng() * 0.14);
        }
      }
    }

    const crownC = vec3(lean * trunkH * 0.08, trunkH * 0.3, 0);
    const crownRx = trunkH * 0.34;
    const crownRy = trunkH * 0.28;
    for (let i = 0; i < VoxelTree.LEAF_CROWN; i++) {
      let p = vec3(randnFrom(rng), randnFrom(rng), randnFrom(rng));
      const len = vLen(p) + 1e-9;
      p = vScale(p, (1 / len) * rng());
      let lpos = vAdd(crownC, [p[0] * crownRx, p[1] * crownRy, p[2] * crownRx]);
      if (lpos[1] < 0) lpos[1] = -lpos[1] * 0.5;
      this._leaf(lpos, 2.2 + rng() * 1.4, 0.8 + rng() * 0.2);
    }
  }

  _rotationMatrix(rotation) {
    const [rx, ry, rz] = rotation;
    const cx = Math.cos(rx), sx = Math.sin(rx);
    const cy = Math.cos(ry), sy = Math.sin(ry);
    const cz = Math.cos(rz), sz = Math.sin(rz);
    const rotX = [
      [1, 0, 0],
      [0, cx, -sx],
      [0, sx, cx],
    ];
    const rotY = [
      [cy, 0, sy],
      [0, 1, 0],
      [-sy, 0, cy],
    ];
    const rotZ = [
      [cz, -sz, 0],
      [sz, cz, 0],
      [0, 0, 1],
    ];
    const zy = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        let s = 0;
        for (let k = 0; k < 3; k++) s += rotZ[i][k] * rotY[k][j];
        zy[i][j] = s;
      }
    }
    const out = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        let s = 0;
        for (let k = 0; k < 3; k++) s += zy[i][k] * rotX[k][j];
        out[i][j] = s;
      }
    }
    return out;
  }

  draw(ctx, W, H, base, rotation, t, dt = 1 / 30) {
    if (!this.active) return;
    let growP = Math.min(this.growTime / VoxelTree.GROW_DURATION, 1);
    let fadeP = 0;
    if (this.growing) {
      this.growTime += dt;
      growP = Math.min(this.growTime / VoxelTree.GROW_DURATION, 1);
      if (growP >= 1) this.growing = false;
    } else if (this.fading) {
      this.fadeTime += dt;
      fadeP = Math.min(this.fadeTime / VoxelTree.FADE_DURATION, 1);
      if (fadeP >= 1) {
        this.active = false;
        return;
      }
    }

    const rot = this._rotationMatrix(rotation);
    const shift = this._trunkH / 2;
    const items = [];

    for (const seg of this.segments) {
      let a = [...seg.a];
      let b = [...seg.b];
      const heightF = clip((a[1] + b[1]) / 2 / VoxelTree.TRUNK_HEIGHT + 0.5, 0, 1.2);
      const sway = VoxelTree.SWAY * heightF * Math.sin(t * VoxelTree.SWAY_WIND + seg.phase);
      a[0] += sway;
      b[0] += sway;
      const ra = this._applyRot(rot, a);
      const rb = this._applyRot(rot, b);
      const tf = clip((growP - seg.reveal) / 0.05, 0, 1);
      if (tf <= 0) continue;
      const rbEff = vLerp(ra, rb, tf);
      const ax = base[0] + ra[0];
      const ay = base[1] - (ra[1] + shift);
      const bx = base[0] + rbEff[0];
      const by = base[1] - (rbEff[1] + shift);
      const r = Math.max(1, Math.round(2 * seg.r));
      if (
        (ax < -r && bx < -r) ||
        (ax > W + r && bx > W + r) ||
        (ay < -r && by < -r) ||
        (ay > H + r && by > H + r)
      ) {
        continue;
      }
      items.push([
        (ra[2] + rbEff[2]) * 0.5,
        "seg",
        ax, ay, bx, by, r, seg.color, 1 - fadeP,
      ]);
    }

    for (const leaf of this.leaves) {
      let pos = [...leaf.pos];
      const heightF = clip(pos[1] / VoxelTree.TRUNK_HEIGHT + 0.5, 0, 1.2);
      if (fadeP > 0) {
        pos[1] -= fadeP * VoxelTree.FALL_DIST * (0.5 + 0.5 * leaf.phase);
        pos[0] += Math.sin(t * 2 + leaf.phase) * fadeP * 3;
      }
      const sway =
        VoxelTree.SWAY * heightF * Math.sin(t * VoxelTree.SWAY_WIND + leaf.phase);
      pos[0] += sway + Math.sin(t * 3.1 + leaf.phase) * 0.6;
      const rp = this._applyRot(rot, pos);
      const sx = base[0] + rp[0];
      const sy = base[1] - (rp[1] + shift);
      if (sx < -10 || sx > W + 10 || sy < -10 || sy > H + 10) continue;
      const alpha = smoothstep((growP - leaf.reveal) / 0.07) * (1 - fadeP);
      if (alpha <= 0) continue;
      items.push([rp[2], "leaf", sx, sy, leaf.r, leaf.color, alpha]);
    }

    items.sort((a, b) => a[0] - b[0]);
    ctx.save();
    ctx.lineCap = "round";
    for (const item of items) {
      if (item[1] === "seg") {
        const [_, kind, x1, y1, x2, y2, r, color, alpha] = item;
        ctx.globalAlpha = alpha;
        ctx.strokeStyle = `rgb(${color[0]},${color[1]},${color[2]})`;
        ctx.lineWidth = r;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      } else {
        const [_, kind, sx, sy, r, color, alpha] = item;
        ctx.globalAlpha = alpha;
        ctx.fillStyle = `rgb(${color[0]},${color[1]},${color[2]})`;
        ctx.beginPath();
        ctx.arc(sx, sy, Math.max(1, r), 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.restore();
  }

  _applyRot(rot, v) {
    return [
      rot[0][0] * v[0] + rot[0][1] * v[1] + rot[0][2] * v[2],
      rot[1][0] * v[0] + rot[1][1] * v[1] + rot[1][2] * v[2],
      rot[2][0] * v[0] + rot[2][1] * v[1] + rot[2][2] * v[2],
    ];
  }
}