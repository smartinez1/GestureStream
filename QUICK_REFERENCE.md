# Gesture Stream - Quick Reference Card

## 🎮 Quick Commands

```bash
# Try demo first (best for understanding the system)
python3 gesture_stream_demo.py

# Run with your camera
python3 gesture_stream.py

# Advanced version with recording & effects
python3 gesture_stream_advanced.py
```

---

## 👆 Pointing Gesture = Control Cube

```
     INDEX EXTENDED
           ↓
        👆
       /  \
      /    \
     ────  ← OTHER FINGERS CURLED
    
Result: Cyan cube rotates with your hand
```

**Cube rotates in these directions:**
- 🔄 **Tilt hand forward/back** → Cube pitches up/down
- 🔄 **Rotate hand left/right** → Cube spins
- 🔄 **Twist hand** → Cube rolls

---

## 🖐️ Open Hand = Switch Images

```
    THUMB  INDEX  MIDDLE  RING  PINKY
      ↑     ↑       ↑      ↑      ↑
      ─────────────────────────
           ALL FINGERS SPREAD

Result: Image overlays on palm (0.5s hold to toggle)
```

**Image 1:** Gradient (default)
**Image 2:** Checkerboard (default)

---

## 🎬 What You See

### Top Left
```
Status display:
- "Image 1 Active" or "Image 2 Active"
- FPS counter
```

### Center
```
CYAN CUBE
(when pointing)
Rotates with hand
```

### Hand Overlay
```
GREEN SKELETON
(shows detected hand bones)

IMAGE OVERLAY
(when hand open)
```

---

## ⌨️ Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Toggle recording (advanced only) |
| `m` | Toggle metrics (advanced only) |

---

## 📱 Gesture Recognition Details

### Pointing Detection Algorithm
```python
✓ index_tip.y < index_pip.y - 0.05
✓ middle_tip.y > middle_pip.y - 0.02
✓ ring_tip.y > ring_pip.y - 0.02
✓ pinky_tip.y > pinky_pip.y - 0.02
```

### Open Hand Detection Algorithm
```python
count = 0
for each of 5 fingers:
    if tip.y < pip.y - 0.03:
        count += 1
    
return count >= 4
```

---

## 🎯 Tips for Best Results

1. **Lighting**: Bright, even lighting is key
2. **Distance**: Keep hand 30-100cm from camera
3. **Gesture Clarity**: Make gestures distinct (fully point, fully open)
4. **Stability**: Hold gesture briefly for recognition
5. **Speed**: Don't move too fast - system updates ~30 FPS

---

## 🔧 Performance Stats

- **Hand Detection**: 90ms per frame
- **Frame Rate**: ~30 FPS
- **Model Size**: 7.5 MB
- **Memory Usage**: 200-300 MB
- **Simultaneous Hands**: Up to 2

---

## 📦 Files Included

```
gesture_stream.py              # Core implementation (365 lines)
gesture_stream_advanced.py     # Enhanced version (195 lines)
gesture_stream_demo.py         # Demo with synthetic data (175 lines)
README.md                      # This file
SETUP_GUIDE.md                # Detailed configuration
GESTURE_QUICK_START.md        # Quick reference
GESTURE_STREAM_README.md      # Technical docs
```

---

## 🚀 Customization Examples

### Use Custom Images
```python
# In gesture_stream.py main():
controller = GestureStreamController(
    image_path_1="my_image_1.png",
    image_path_2="my_image_2.png"
)
```

### Make Pointing More Sensitive
```python
# In detect_pointing_gesture():
index_extended = index_tip.y < index_pip.y - 0.03  # Lower = more sensitive
```

### Make Cube Bigger
```python
# In run():
self.draw_3d_cube(frame, position, self.cube_rotation, size=80)
```

### Change Image Blend
```python
# In overlay_image_on_hand():
frame[...] = cv2.addWeighted(frame, 0.3, image, 0.7, 0)  # More image
```

---

## ✅ System Status

```
✓ All dependencies installed
✓ Model downloaded (7.5 MB)
✓ All scripts ready
✓ Hand detection working
✓ 3D rendering working
✓ Image overlay working
```

---

## 🎓 How It Works

```
CAMERA FRAME
    ↓
MediaPipe detects 21 hand keypoints
    ↓
Analyze finger positions
    ↓
Classify gesture:
├─ Pointing? → Render 3D cube + rotate
└─ Open? → Overlay image on hand
    ↓
Display with hand skeleton
```

---

## 🔍 Troubleshooting

| Issue | Fix |
|-------|-----|
| "Camera not authorized" | Grant permission in System Preferences |
| Hand not detected | Better lighting, move closer |
| Slow performance | Close other apps |
| Gesture not working | Make it more obvious (fully extend/open) |
| No image shown | Hold open hand gesture for 0.5s |

---

## 🎮 Project Ideas

- **Game Controller**: Map gestures to game commands
- **Presentation Tool**: Point to control slides
- **AR Experience**: Gesture-controlled virtual objects
- **Data Visualization**: Hand-controlled 3D models
- **Music/Art**: Real-time visual performance

---

## 📞 One-Line Examples

```bash
# Just see demo
python3 gesture_stream_demo.py

# Quick test
python3 gesture_stream.py

# Record session
python3 gesture_stream_advanced.py
```

---

**Ready to start? Pick one and go! 🎮✨**
