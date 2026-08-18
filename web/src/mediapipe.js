import { FilesetResolver, HandLandmarker, ImageSegmenter } from "@mediapipe/tasks-vision";

const WASM =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/" +
  "hand_landmarker/float16/1/hand_landmarker.task";
const SEG_MODEL =
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/" +
  "selfie_segmenter/float16/latest/selfie_segmenter.tflite";

export async function initHandLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks(WASM);
  return HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: HAND_MODEL },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}

export async function initImageSegmenter() {
  const fileset = await FilesetResolver.forVisionTasks(WASM);
  return ImageSegmenter.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: SEG_MODEL },
    runningMode: "VIDEO",
    outputCategoryMask: true,
    outputConfidenceMasks: true,
  });
}