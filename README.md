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
python3 core/gesture_stream_demo.py
\`\`\`

See animated demonstration of all features.

### Run Live with Camera

\`\`\`bash
python3 core/gesture_stream.py
\`\`\`

**Controls:**
- 👆 Point with index finger → Control rotating 3D cube
- 🖐️ Open hand flat → Toggle between images
- \`q\` → Quit

### Run Advanced Version

\`\`\`bash
python3 core/gesture_stream_advanced.py
\`\`\`

**Additional Controls:**
- \`r\` → Start/stop video recording
- \`m\` → Toggle metrics display

### Run All Effects Together (Showcase)

\`\`\`bash
python3 gesture_stream_showcase.py          # camera
python3 gesture_stream_showcase.py --demo   # synthetic demo, no camera
\`\`\`

Runs the fire, tree, kaleidoscope and color-trail effects simultaneously:

- 🤟 L sign (index + thumb out) → fire burns at that palm (raise/lower your other hand to control the heat: top of the frame = biggest fire; the flame fades in/out smoothly and survives brief tracking flicker)
- 🙌 Hands spread apart, then together with palms up → a tree grows between the palms
- 🙏 Hold a prayer pose (0.6s) → toggle kaleidoscope
- 👋 Wave with both hands (3 swings) → toggle color trail (blues and purples)
- `h` → Toggle HUD, `q` → Quit

### Web App (Browser, Vercel-Deployable)

```bash
cd web
npm install
npm run dev        # camera demo at http://localhost:5173
npm run build      # outputs web/dist (what Vercel serves)
```

Append `?demo=1` to the URL for the synthetic no-camera demo (4-phase 36 s cycle: fire → tree → kaleido → trail). The web app runs fully in-browser via the MediaPipe Tasks API (`.wasm`) — no Python, no backend. It lives on the `web-app` branch; the root `vercel.json` builds `web/` and serves `web/dist`. Same gestures and effects as the showcase, ported 1:1 to Canvas2D (`web/src/gestures/*.js` mirror the Python state classes).

### Standalone Effect Scripts

\`\`\`bash
python3 effects/gesture_stream_fire.py --demo      # fire
python3 effects/gesture_stream_tree.py --demo      # procedural tree
python3 effects/gesture_stream_effects.py --demo   # kaleidoscope + color trail
\`\`\`

Each also runs live with the camera (drop \`--demo\`).

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
├── gesture_stream_showcase.py          # All four effects combined
├── web/                                # Web app (Canvas2D port, branch web-app)
│   ├── index.html                      # Canvas + status bar
│   └── src/                            # Vite + MediaPipe Tasks API
│       ├── main.js                     # Render loop, demo state machine
│       ├── mediapipe.js                # Model init wrapper
│       ├── gestures/                   # Gesture state classes (1:1 with Python)
│       └── effects/                    # Fire, tree, kaleidoscope, color trail
├── vercel.json                         # Deploys web/ on Vercel
├── core/
│   ├── gesture_stream.py               # Core implementation
│   ├── gesture_stream_advanced.py      # Enhanced version with effects
│   ├── gesture_stream_demo.py          # Demo with synthetic data
│   ├── gesture_stream_diffusion.py     # Voxel diffusion module
│   └── __init__.py
├── effects/
│   ├── gesture_stream_fire.py          # Fire effect
│   ├── gesture_stream_tree.py          # Procedural tree effect
│   ├── gesture_stream_effects.py       # Kaleidoscope + color trail
│   └── __init__.py
├── images/                             # Images projected on the cube
├── README.md                           # This file
├── docs/                               # Detailed documentation
│   ├── QUICK_REFERENCE.md              # Quick commands & tips
│   └── SETUP_GUIDE.md                  # Detailed configuration
├── requirements.txt                    # Python dependencies
└── .gitignore                          # Git ignore rules
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
from core.gesture_stream import GestureStreamController

controller = GestureStreamController()
controller.run()
\`\`\`

## 📚 Documentation

- **QUICK_REFERENCE.md** - Quick commands and tips (in `docs/`)
- **SETUP_GUIDE.md** - Detailed configuration (in `docs/`)
- **GESTURE_STREAM_README.md** - Technical details (in `docs/`)

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
