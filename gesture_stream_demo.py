"""
Gesture Stream Demo - Simulated version for testing without camera
Shows how the system works with synthetic hand data
"""

import cv2
import numpy as np
import time
from gesture_stream import GestureStreamController


class GestureStreamDemo(GestureStreamController):
    """Demo version that simulates hand gestures."""

    def run_demo(self):
        """Run demo with synthetic hand data."""
        print("Gesture Stream Demo (No Camera Required)")
        print("=" * 50)
        print("Simulating hand gestures...")
        print("Press 'q' to quit\n")

        frame_count = 0
        demo_time = 0
        h, w = 720, 1280

        while True:
            frame = np.zeros((h, w, 3), dtype=np.uint8)

            # Create gradient background
            for i in range(w):
                frame[:, i] = [int(i / w * 50), int(i / w * 100), 100]

            # Simulate pointing gesture with moving hand
            if demo_time < 5:
                # Pointing gesture - cube rotates
                x = int(w / 2 + 200 * np.cos(demo_time))
                y = int(h / 2 + 150 * np.sin(demo_time * 0.5))

                # Simulate rotation based on time
                rotation = np.array([
                    np.sin(demo_time) * 0.5,
                    np.cos(demo_time) * 0.5,
                    np.sin(demo_time * 0.3) * 0.3
                ])

                self.draw_3d_cube(frame, (x, y), rotation, size=40)

                cv2.putText(frame, "POINTING GESTURE", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Cube Position: ({x}, {y})",
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           (200, 200, 0), 1)

            elif demo_time < 10:
                # Open hand gesture - image overlay
                x = int(w / 3)
                y = int(h / 2)

                frame = self.overlay_image_on_hand(
                    frame, self.image_1, (x, y), scale=0.4
                )

                cv2.putText(frame, "OPEN HAND GESTURE", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 200), 2)
                cv2.putText(frame, f"Image Position: ({x}, {y})",
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           (200, 200, 0), 1)
                cv2.putText(frame, "Image 1: Gradient", (10, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 100), 1)

            elif demo_time < 15:
                # Open hand gesture with image 2
                x = int(2 * w / 3)
                y = int(h / 2)

                frame = self.overlay_image_on_hand(
                    frame, self.image_2, (x, y), scale=0.4
                )

                cv2.putText(frame, "IMAGE SWITCH", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 100, 255), 2)
                cv2.putText(frame, f"Image Position: ({x}, {y})",
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           (200, 200, 0), 1)
                cv2.putText(frame, "Image 2: Checkerboard", (10, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 100), 1)

            # Reset animation
            if demo_time >= 15:
                demo_time = 0

            # Status panel
            cv2.rectangle(frame, (10, h - 100), (400, h - 10), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, h - 100), (400, h - 10), (0, 255, 0), 2)

            cv2.putText(frame, "FEATURES:", (20, h - 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            cv2.putText(frame, "✓ Hand detection", (20, h - 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
            cv2.putText(frame, "✓ 3D cube control", (20, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)

            cv2.putText(frame, "✓ Image overlay", (220, h - 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
            cv2.putText(frame, "✓ Gesture recognition", (220, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)

            # Frame counter and time
            cv2.putText(frame, f"Demo Time: {demo_time:.1f}s | Frame: {frame_count}",
                       (w - 400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (100, 200, 255), 1)

            cv2.imshow("Gesture Stream Demo", frame)

            key = cv2.waitKey(33) & 0xFF  # ~30 FPS
            if key == ord("q"):
                break

            frame_count += 1
            demo_time += 0.033

        cv2.destroyAllWindows()


def main():
    """Run the demo."""
    demo = GestureStreamDemo()
    demo.run_demo()


if __name__ == "__main__":
    main()
