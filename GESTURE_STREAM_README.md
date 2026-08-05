# Gesture Stream Controller

Real-time gesture-based interactive stream that uses hand tracking to control a 3D cube and swap images.

## Features

✨ **Hand Gesture Recognition**
- Detects pointing gesture (easily detectable, index finger extended)
- Detects open hand gesture (all fingers extended)

🎲 **Interactive 3D Cube**
- Cube orientation follows hand pointing direction
- Real-time rotation based on hand pose estimation
- Smooth rotation history for stable control

🖼️ **Image Overlay System**
- Display images on gesture hand
- Switch between two images using open hand gesture
- Blend images with hand position for immersive effect

## Requirements

- Python 3.7+
- OpenCV (cv2)
- MediaPipe
- NumPy
- Pillow

## Installation

```bash
pip install mediapipe opencv-python pillow numpy
```

## Usage

### Basic Usage

```bash
python3 gesture_stream.py
```

### With Custom Images

Edit the `main()` function to use your own images:

```python
controller = GestureStreamController(
    image_path_1="path/to/image1.png",
    image_path_2="path/to/image2.png",
)
```

## How It Works

### Gestures

1. **Pointing Gesture** (Primary Control)
   - Extend index finger, curl other fingers
   - Draws a 3D cube at your fingertip
   - Cube rotates based on hand orientation
   - Perfect for intuitive 3D control

2. **Open Hand Gesture** (Image Switching)
   - Keep hand flat with all fingers extended
   - Toggles between Image 1 and Image 2
   - Image is overlaid on your hand
   - Blended with the video stream for visual effect

### Gesture Detection Algorithm

**Pointing Detection:**
- Index finger tip below PIP joint (extended)
- Other fingers (middle, ring, pinky) above their PIP joints (curled)
- Ensures clear, unambiguous pointing

**Open Hand Detection:**
- At least 4 of 5 fingers extended
- All fingertips below their PIP joints
- Smooth, flat palm orientation

### 3D Cube Rotation

Orientation calculated from:
- **Pitch**: Forward tilt (wrist to middle finger)
- **Yaw**: Side rotation (hand rotation about vertical axis)
- **Roll**: Palm twist (thumb to finger spread)

Rotation history (last 5 frames) smooths jitter for stable visualization.

## Controls

- **Point with hand**: Control cube rotation
- **Open hand**: Switch active image
- **'q' key**: Quit application

## Customization

### Adjust Gesture Sensitivity

In `gesture_stream.py`, modify detection thresholds:

```python
# More sensitive pointing detection (lower = more sensitive)
index_extended = index_tip.y < index_pip.y - 0.03  # Change 0.05

# More sensitive open hand detection
if tip.y < pip.y - 0.02:  # Change 0.03
```

### Change Cube Size

```python
self.draw_3d_cube(frame, position, self.cube_rotation, size=60)  # Adjust size
```

### Adjust Image Overlay Scale

```python
self.overlay_image_on_hand(frame, self.current_image, position, scale=0.4)  # 0.3 default
```

## Architecture

**GestureStreamController**
- `detect_pointing_gesture()`: Identifies pointing hand pose
- `detect_open_hand_gesture()`: Identifies open hand pose
- `calculate_hand_orientation()`: Extracts 3D rotation from hand landmarks
- `draw_3d_cube()`: Renders rotating cube with perspective
- `overlay_image_on_hand()`: Blends image onto video stream
- `process_frame()`: Main detection pipeline

## Performance Notes

- Runs at ~30 FPS on modern hardware
- MediaPipe hand detection: ~90ms per frame
- Optimized for 1280x720 resolution
- Smoothing via rotation history reduces jitter

## Troubleshooting

**Hand not detected:**
- Ensure good lighting
- Keep hand clearly visible in frame
- Try adjusting `min_detection_confidence` (0.7) lower

**Gesture not recognized:**
- Make pointing gesture more distinct (index extended further)
- For open hand, spread fingers wide apart

**Slow performance:**
- Reduce frame resolution
- Lower detection confidence threshold
- Close other applications

## Future Enhancements

- Multi-hand gesture combinations
- Gesture recording/playback
- ML model for custom gestures
- Hand pose classification network
- Depth-based occlusion handling
- Haptic feedback simulation
