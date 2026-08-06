# Gesture Stream Quick Start

## Installation

All dependencies are already installed. Just run:

```bash
python3 gesture_stream.py
```

## What You Get

### ✨ Basic Version: `gesture_stream.py`
- Real-time hand tracking with MediaPipe
- **Pointing gesture**: Index finger extended controls a 3D cube
  - Cube rotates based on your hand orientation
  - Hand pitch (up/down), yaw (left/right), and roll (rotation) all tracked
- **Open hand gesture**: All fingers extended
  - Toggles between Image 1 (gradient) and Image 2 (checkerboard)
  - Image overlays on your hand with 50% blend

### 🎨 Advanced Version: `gesture_stream_advanced.py`
All basic features PLUS:
- **Gesture trail**: See the path your pointing finger takes
- **Status panel**: Real-time gesture recognition feedback
- **Cube glow effect**: Visual feedback for active gesture
- **Rotation indicators**: Arrow showing current cube orientation
- **Recording**: Press 'r' to record the stream as MP4
- **Metrics display**: Press 'm' to show hand position metrics

## How to Use

### Basic Stream
```bash
python3 gesture_stream.py
```
Then:
1. **Point with index finger** → see the cube appear and rotate with your hand
2. **Open your hand flat** → toggle between images on your palm
3. **Press 'q'** → quit

### Advanced Stream
```bash
python3 gesture_stream_advanced.py
```
Controls:
- **Point**: Control cube (same as basic)
- **Open hand**: Switch images (same as basic)
- **'r'**: Start/stop recording
- **'m'**: Toggle metrics overlay
- **'q'**: Quit

## Custom Images

Edit the `main()` function in either script:

```python
# Change these paths to use your own images
stream = GestureStreamController(
    image_path_1="path/to/your/image1.png",
    image_path_2="path/to/your/image2.png"
)
```

## Gesture Sensitivity

Adjust detection thresholds in the script:

**Pointing gesture** (more sensitive = lower threshold):
```python
index_extended = index_tip.y < index_pip.y - 0.05  # Default: 0.05
```

**Open hand gesture** (more sensitive = higher threshold):
```python
if tip.y < pip.y - 0.03:  # Default: 0.03
    extended_count += 1
```

## Output Files

When recording with advanced version:
- Videos save to `gesture_recordings_YYYYMMDD_HHMMSS/` folder
- File: `gesture_stream.mp4`

## Troubleshooting

**Hand not detected?**
- Ensure good lighting
- Keep hand clearly visible (not too close or too far)
- Try different hand angles

**Gesture not recognized?**
- Point more distinctly (extend index finger further)
- For open hand, spread all fingers wide apart
- Hold gesture steady for a moment

**Poor performance?**
- Check if other apps are using the camera
- Close other applications
- Try reducing window size

## What's Happening Under the Hood

1. **MediaPipe HandLandmarker**: Detects 21 hand keypoints in real-time
2. **Gesture Classification**: 
   - Analyzes relative finger positions to classify gesture type
   - Uses position history for stability
3. **3D Rotation Calculation**:
   - Extracts pitch (forward tilt), yaw (horizontal rotation), roll (twist)
   - Uses palm normal vector and finger positions
4. **3D Rendering**: Projects 3D cube vertices onto 2D screen with proper perspective
5. **Image Blending**: 50/50 alpha blend overlays image onto video stream

Enjoy your gesture-controlled interactive stream! 🎮
