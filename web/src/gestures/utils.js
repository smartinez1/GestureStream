export function mean(xs) {
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

export function palmCenter(lm) {
  return {
    x: (lm[0].x + lm[5].x + lm[17].x) / 3,
    y: (lm[0].y + lm[5].y + lm[17].y) / 3,
  };
}

export function dist2d(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

export function isOpen(lm, offset = 0.03) {
  let extended = 0;
  const pairs = [[4, 3], [8, 6], [12, 10], [16, 14], [20, 18]];
  for (const [tip, pip] of pairs) {
    if (lm[tip].y < lm[pip].y - offset) extended += 1;
  }
  return extended >= 4;
}

export function pointsUp(lm, threshold = 1.0) {
  const wrist = lm[0];
  const mcp = lm[9];
  const tip = lm[12];
  const palm = Math.max(dist2d(mcp, wrist), 1e-6);
  return (tip.y - wrist.y) / palm < -threshold;
}

export function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

export function norm3(v) {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

export function clip(v, lo, hi) {
  return Math.min(Math.max(v, lo), hi);
}

export function smoothstep(t) {
  t = clip(t, 0, 1);
  return t * t * (3 - 2 * t);
}

export function lerp(a, b, f) {
  return a + (b - a) * f;
}

export function randn() {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}