# GestureStream

Real-time gesture-based interactive 3D control system using hand tracking and MediaPipe.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎮 Features

- ✨ **Real-time hand detection** with 21 keypoints per hand
- 👆 **Pointing gesture** controls a rotating 3D cube
- 🖐️ **Open hand gesture** overlays and switches images
- 🔄 **3D rotation** based on hand orientation (pitch, yaw, roll)
- 📸 **Image overlay** with 50% alpha blending
- 🎥 **Video recording** capability (advanced version)
- 📊 **Live metrics** display (advanced version)
- 🎨 **Gesture visualization** with trails (advanced version)

## 🚀 Quick Start

### Installation

\`\`\`bash
# Clone the repository
git clone https://github.com/yourusername/GestureStream.git
cd GestureStream

# Install dependencies
pip install -r requirements.txt
\`\`\`

### Run Demo (No Camera Required)

\`\`\`bash
python3 gesture_stream_demo.py
\`\`\`

See animated demonstration of all features.

### Run Live with Camera

\`\`\`bash
python3 gesture_stream.py
\`\`\`

**Controls:**
- 👆 Point with index finger → Control rotating 3D cube
- 🖐️ Open hand flat → Toggle between images
- \`q\` → Quit

### Run Advanced Version

\`\`\`bash
python3 gesture_stream_advanced.py
\`\`\`

**Additional Controls:**
- \`r\` → Start/stop video recording
- \`m\` → Toggle metrics display

## 📋 Requirements

- Python 3.7+
- OpenCV (cv2)
- MediaPipe 1.0.0
- NumPy
- Pillow

All dependencies listed in \`requirements.txt\`

## 📁 Project Structure

\`\`\`
GestureStream/
├── gesture_stream.py              # Core implementation
├── gesture_stream_advanced.py     # Enhanced version with effects
├── gesture_stream_demo.py         # Demo with synthetic data
├── README.md                      # This file
├── QUICK_REFERENCE.md            # Quick commands & tips
├── SETUP_GUIDE.md                # Detailed configuration
├── requirements.txt               # Python dependencies
└── .gitignore                     # Git ignore rules
\`\`\`

## 👆 Gesture Recognition

### Pointing Gesture

Extend your **index finger** while curling other fingers.

\`\`\`
INDEX EXTENDED + OTHER FINGERS CURLED
         ↓
   CYAN CUBE APPEARS
         ↓
   Rotates with hand:
   • Tilt = Pitch
   • Rotate = Yaw
   • Twist = Roll
\`\`\`

### Open Hand Gesture

Spread all **5 fingers** flat.

\`\`\`
ALL FINGERS EXTENDED
         ↓
   IMAGE OVERLAYS ON PALM
         ↓
   Hold 0.5s to TOGGLE IMAGE
\`\`\`

## 🎯 How It Works

### Detection Pipeline

\`\`\`
Camera Frame
    ↓
MediaPipe HandLandmarker (21 keypoints per hand)
    ↓
Gesture Classification
    ├─ Pointing: Index extended, others curled
    └─ Open Hand: ≥4 fingers extended
    ↓
3D Hand Orientation Calculation
    ↓
Render 3D Cube OR Overlay Image
\`\`\`

## 🔧 Customization

### Use Custom Images

Edit the \`main()\` function:

\`\`\`python
controller = GestureStreamController(
    image_path_1="path/to/image1.png",
    image_path_2="path/to/image2.png"
)
\`\`\`

### Adjust Gesture Sensitivity

In \`detect_pointing_gesture()\`:

\`\`\`python
# More sensitive (lower = more sensitive)
index_extended = index_tip.y < index_pip.y - 0.03  # Default: 0.05
\`\`\`

## 📊 Performance

| Metric | Value |
|--------|-------|
| Model Size | 7.5 MB |
| Inference Time | ~90ms per frame |
| Frame Rate | ~30 FPS |
| Memory Usage | 200-300 MB |
| Max Hands | 2 simultaneous |

## 🎓 Quick API

\`\`\`python
from gesture_stream import GestureStreamController

controller = GestureStreamController()
controller.run()
\`\`\`

## 📚 Documentation

- **QUICK_REFERENCE.md** - Quick commands and tips
- **SETUP_GUIDE.md** - Detailed configuration
- **GESTURE_STREAM_README.md** - Technical details

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not authorized | Grant Terminal camera permission |
| Hand not detected | Improve lighting, move closer |
| Poor performance | Close other apps |

## 💡 Use Cases

- 🎮 Game control
- 📊 Data visualization
- 🎨 Creative performance
- 🎓 Educational demos
- 📱 AR applications

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- MediaPipe for hand detection
- OpenCV for computer vision
- NumPy for numerical computing

---

**Made with ❤️ for gesture-based interaction**
