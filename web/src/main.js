import { initHandLandmarker, initImageSegmenter } from "./mediapipe.js";
import { FireGestureState } from "./gestures/fireGesture.js";
import { TreeGestureState } from "./gestures/treeGesture.js";
import { KaleidoscopeGestureState } from "./gestures/kaleidoGesture.js";
import { WaveGestureState, PersonSegmenter, PersonTrackerState } from "./gestures/waveGesture.js";
import { FireEffect } from "./effects/fire.js";
import { VoxelTree } from "./effects/tree.js";
import { Kaleidoscope } from "./effects/kaleidoscope.js";
import { ColorDiffusion } from "./effects/colorDiffusion.js";

const DEMO = new URLSearchParams(location.search).has("demo");

const video = document.getElementById("cam");
const canvas = document.getElementById("out");
const ctx = canvas.getContext("2d");
const loading = document.getElementById("loading");
const statusEls = {
  fire: document.getElementById("eff-fire"),
  tree: document.getElementById("eff-tree"),
  kaleido: document.getElementById("eff-kaleido"),
  trail: document.getElementById("eff-trail"),
};
const fpsEl = document.getElementById("fps");

let W = 0;
let H = 0;

const fireGesture = new FireGestureState();
const treeGesture = new TreeGestureState();
const kaleidoGesture = new KaleidoscopeGestureState();
const waveGesture = new WaveGestureState();
const fire = new FireEffect();
const tree = new VoxelTree();
const kaleido = new Kaleidoscope();
const diffusion = new ColorDiffusion();

let handLandmarker = null;
let segmenter = null;
let personSegmenter = null;
let personTracker = null;
let showHud = true;

const demoState = DEMO ? createDemoState() : null;

function resize() {
  W = window.innerWidth;
  H = window.innerHeight;
  canvas.width = W;
  canvas.height = H;
}
window.addEventListener("resize", resize);
resize();

async function init() {
  if (DEMO) {
    demoState.videoCv = makeDemoVideo();
    loading.remove();
    requestAnimationFrame(renderFrame);
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
    });
    video.srcObject = stream;
    await video.play();
  } catch (e) {
    loading.textContent = "Camera unavailable — open with ?demo=1 for the synthetic demo";
    return;
  }
  try {
    loading.textContent = "Loading MediaPipe models…";
    handLandmarker = await initHandLandmarker();
    segmenter = await initImageSegmenter();
    personSegmenter = new PersonSegmenter(segmenter);
    personTracker = new PersonTrackerState(personSegmenter);
  } catch (e) {
    console.error("MediaPipe init failed", e);
    loading.textContent = "Failed to load MediaPipe models";
    return;
  }
  loading.remove();
  requestAnimationFrame(renderFrame);
}

let lastTs = null;
let frameCount = 0;
let fpsWindowStart = 0;
let fps = 0;

function renderFrame(ts) {
  const dt = lastTs === null ? 1 / 30 : Math.min((ts - lastTs) / 1000, 0.1);
  lastTs = ts;
  const now = ts / 1000;

  frameCount += 1;
  if (fpsWindowStart === 0) fpsWindowStart = ts;
  if (frameCount % 30 === 0) {
    fps = 30000 / (ts - fpsWindowStart);
    fpsWindowStart = ts;
    fpsEl.textContent = `FPS: ${fps.toFixed(0)}`;
  }

  let hands = [];
  let handedness = [];
  const src = DEMO ? demoState.videoCv : video;

  if (DEMO) {
    [hands, handedness] = demoState.hands(now);
  } else {
    ctx.drawImage(video, 0, 0, W, H);
    if (handLandmarker !== null) {
      const result = handLandmarker.detectForVideo(video, ts);
      hands = result.handLandmarks;
      handedness = result.handedness.map((h) => h[0].categoryName);
    }
    if (personTracker !== null) personTracker.update(video, W, H);
  }

  let maskCv = null;
  let personBox = null;
  if (DEMO) {
    const [box, mask] = demoState.person();
    personBox = box;
    maskCv = mask;
  } else if (personTracker !== null) {
    personBox = personTracker.personRect;
    if (personSegmenter !== null) maskCv = personSegmenter.mask;
  }

  const [triggerCenter, heat, burning] = fireGesture.update(hands, [H, W]);
  if (burning) fire.ignite();
  else fire.extinguish();

  treeGesture.update(hands, handedness, [H, W], now, tree);
  kaleidoGesture.update(hands, now);
  waveGesture.update(hands, now);

  // a. kaleidoscope
  if (kaleidoGesture.enabled || kaleido.blend > 0.01) {
    kaleido.apply(
      ctx,
      src,
      W,
      H,
      dt,
      personBox,
      personTracker !== null ? personTracker.center : null,
      kaleidoGesture.enabled
    );
  }

  // b. color trail
  if (waveGesture.waveEnabled || diffusion.isActive()) {
    diffusion.apply(ctx, maskCv, now, dt, waveGesture.waveEnabled, W, H);
  }

  // c. tree
  if (tree.isActive()) {
    const rotation = [
      0.22 * Math.sin(now * 0.7),
      0.3 * Math.cos(now * 0.5),
      0.15 * Math.sin(now * 0.9),
    ];
    tree.draw(
      ctx,
      W,
      H,
      [treeGesture.anchor[0], treeGesture.anchor[1] - TreeGestureState.ANCHOR_UP_OFFSET],
      rotation,
      now,
      dt
    );
  }

  // d. fire
  if (triggerCenter !== null) {
    fire.draw(ctx, W, H, triggerCenter, heat * fireGesture.flame, dt);
  }

  if (showHud) drawHud(hands.length);

  updateStatusBar();
  requestAnimationFrame(renderFrame);
}

function drawHud(numHands) {
  const panel = {
    fire: fire.isBurning() ? "BURNING" : "READY",
    intensity: Math.round(fireGesture.heat * fireGesture.flame * 100),
    tree: treeStateText(),
    palmsUp: treeGesture.palmsUp ? "YES" : "NO",
    kaleido: kaleidoGesture.enabled ? "ON" : "OFF",
    trail: waveGesture.waveEnabled ? "ON" : "OFF",
    hands: numHands,
  };
  ctx.save();
  ctx.fillStyle = "rgba(20,20,20,0.7)";
  ctx.fillRect(10, 10, 320, 150);
  ctx.strokeStyle = "rgba(0,255,0,0.8)";
  ctx.strokeRect(10.5, 10.5, 320, 150);
  ctx.fillStyle = "#fff";
  ctx.font = "15px system-ui";
  ctx.textBaseline = "top";
  const lines = [
    `Fire: ${panel.fire} | Intensity: ${panel.intensity}%`,
    `Tree: ${panel.tree} | Palms up: ${panel.palmsUp}`,
    `Kaleidoscope: ${panel.kaleido}`,
    `Color trail: ${panel.trail}`,
    `Hands: ${panel.hands} | H: toggle HUD`,
  ];
  lines.forEach((line, i) => ctx.fillText(line, 22, 24 + i * 26));
  ctx.restore();
}

function treeStateText() {
  if (!tree.isActive()) return "IDLE";
  if (tree.growing) return "GROWING";
  if (tree.fading) return "FADING";
  return "HELD";
}

function updateStatusBar() {
  statusEls.fire.classList.toggle("on", fire.isBurning());
  statusEls.tree.classList.toggle("on", tree.isActive());
  statusEls.kaleido.classList.toggle("on", kaleidoGesture.enabled);
  statusEls.trail.classList.toggle("on", waveGesture.waveEnabled);
}

window.addEventListener("keydown", (e) => {
  if (e.key === "h" || e.key === "H") showHud = !showHud;
});

init();

// ---------------- synthetic demo ----------------

function createDemoState() {
  return {
    videoCv: makeDemoVideo(),
    t0: performance.now() / 1000,
    hands(now) {
      const t = now - this.t0;
      const phase = Math.floor(t / 9) % 4;
      const local = t % 9;
      if (phase === 0) {
        // fire: L sign + modulator hand bobbing (heat)
        const wy = 0.75 - 0.55 * Math.max(0, Math.sin(local * 1.4));
        return [
          [synHand(0.3, "l_sign", "flat", -1, 0.5), synHand(0.7, "open", "flat", 1, wy)],
          ["Left", "Right"],
        ];
      }
      if (phase === 1) {
        // tree: palms-up spread -> together -> hold -> apart
        let x1, x2;
        if (local < 2) {
          x1 = 0.3 + 0.02 * Math.sin(local * 3);
          x2 = 0.7 - 0.02 * Math.sin(local * 3);
        } else if (local < 4.2) {
          const f = (local - 2) / 2.2;
          x1 = 0.3 + 0.17 * f;
          x2 = 0.7 - 0.17 * f;
        } else if (local < 6.5) {
          x1 = 0.47 + 0.01 * Math.sin(local * 2);
          x2 = 0.53 - 0.01 * Math.sin(local * 2);
        } else {
          const f = (local - 6.5) / 2.5;
          x1 = 0.47 - 0.17 * f;
          x2 = 0.53 + 0.17 * f;
        }
        return [
          [synHand(x1, "open", "up", -1, 0.72), synHand(x2, "open", "up", 1, 0.72)],
          ["Right", "Left"],
        ];
      }
      if (phase === 2) {
        // kaleido: prayer pose holds
        const hold = local % 4 < 1.8;
        if (hold) {
          return [prayerHands(), ["Left", "Right"]];
        }
        return [[], []];
      }
      // wave: both hands swing together
      const swing = 0.14 * Math.sin(local * 2 * Math.PI * 1.1);
      return [
        [
          synHand(0.5 + swing, "open", "flat", -1, 0.5),
          synHand(0.5 + swing, "open", "flat", 1, 0.55),
        ],
        ["Left", "Right"],
      ];
    },
    person() {
      const t = (performance.now() / 1000 - this.t0) * 0.4;
      const px = W * (0.5 + 0.15 * Math.sin(t));
      const py = H * (0.5 + 0.08 * Math.sin(t * 0.8));
      const pw = W * 0.24;
      const ph = H * 0.5;
      const box = [px - pw / 2, py - ph / 2, px + pw / 2, py + ph / 2];
      const mw = W >> 1;
      const mh = H >> 1;
      const cv = document.createElement("canvas");
      cv.width = mw;
      cv.height = mh;
      const mctx = cv.getContext("2d", { willReadFrequently: true });
      mctx.fillStyle = "#fff";
      mctx.beginPath();
      mctx.ellipse(mw / 2, mh / 2, (pw / 2) * 0.55, (ph / 2) * 0.9, 0, 0, Math.PI * 2);
      mctx.fill();
      return [box, cv];
    },
  };
}

function makeDemoVideo() {
  const cv = document.createElement("canvas");
  cv.width = 640;
  cv.height = 360;
  const g = cv.getContext("2d");
  const grad = g.createLinearGradient(0, 0, 0, 360);
  grad.addColorStop(0, "#1a2340");
  grad.addColorStop(1, "#0a0e1c");
  g.fillStyle = grad;
  g.fillRect(0, 0, 640, 360);
  g.fillStyle = "rgba(255,255,255,0.04)";
  for (let i = 0; i < 40; i++) {
    g.beginPath();
    g.arc(Math.random() * 640, Math.random() * 360, Math.random() * 2, 0, Math.PI * 2);
    g.fill();
  }
  return cv;
}

function synHand(wx, pose, zmode, side, wy = 0.8) {
  // 21-landmark hand, same template as the Python synthetic harness
  const dy = wy - 0.8;
  const z = zmode === "vertical" ? 0.02 : 0;
  const z9 = zmode === "up" ? -0.1 : zmode === "down" ? 0.1 : z;
  const z5 = zmode === "up" ? -0.09 : zmode === "down" ? 0.09 : z - 0.01;
  const z17 = zmode === "up" ? -0.11 : zmode === "down" ? 0.11 : z + 0.01;
  const s = side;
  const fingersOut = pose === "open";
  const thumbExt = pose === "l_sign" || pose === "open";
  const mTipY = fingersOut ? 0.47 : 0.61;
  const rTipY = fingersOut ? 0.49 : 0.62;
  const pTipY = fingersOut ? 0.51 : 0.63;
  const tTipX = wx + (thumbExt ? 0.105 : 0.03) * s;
  const tTipY = thumbExt ? 0.705 : 0.7;
  const lm = [
    [wx, 0.8, 0],
    [wx + 0.02 * s, 0.76, z],
    [wx + 0.045 * s, 0.745, z],
    [wx + 0.075 * s, 0.72, z],
    [tTipX, tTipY, z],
    [wx + 0.02 * s, 0.69, z5],
    [wx + 0.02 * s, 0.615, z],
    [wx + 0.02 * s, 0.55, z],
    [wx + 0.02 * s, 0.49, z],
    [wx, 0.68, z9],
    [wx, 0.6, z],
    [wx, 0.53, z],
    [wx, mTipY, z],
    [wx - 0.02 * s, 0.69, z],
    [wx - 0.02 * s, 0.615, z],
    [wx - 0.02 * s, 0.55, z],
    [wx - 0.02 * s, rTipY, z],
    [wx - 0.045 * s, 0.7, z17],
    [wx - 0.045 * s, 0.625, z],
    [wx - 0.045 * s, 0.565, z],
    [wx - 0.045 * s, pTipY, z],
  ].map((p) => ({ x: p[0], y: p[1] + dy, z: p[2] }));
  return lm;
}

function prayerHands() {
  const a = synHand(0.49, "open", "vertical", -1);
  const b = synHand(0.51, "open", "vertical", 1);
  a[4].x = b[4].x = 0.5;
  return [a, b];
}