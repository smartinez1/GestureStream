"""
Advanced Gesture Stream with Recording and Effects
Includes gesture recording, visualization, and custom image support
"""

import cv2
import numpy as np
from gesture_stream import GestureStreamController
from datetime import datetime
import os
import time as time_module


class AdvancedGestureStream(GestureStreamController):
    """Extended version with recording and visual effects."""

    def __init__(self, image_path_1=None, image_path_2=None, enable_recording=False):
        super().__init__(image_path_1, image_path_2)

        self.enable_recording = enable_recording
        self.recording = False
        self.gesture_history = []
        self.frame_count = 0
        self.toggle_debounce = 0

        if enable_recording:
            self.setup_recording()

    def setup_recording(self):
        """Setup video recording."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"gesture_recordings_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Recording will be saved to: {self.output_dir}")

    def draw_gesture_trail(self, frame, max_points=20):
        """Draw trail of hand positions."""
        if len(self.gesture_history) < 2:
            return frame

        points = [(int(pos[0]), int(pos[1])) for pos in self.gesture_history[-max_points:]]

        for i in range(1, len(points)):
            alpha = i / len(points)  # Fade effect
            color = (int(255 * alpha), 100, int(255 * (1 - alpha)))
            thickness = max(1, int(3 * alpha))
            cv2.line(frame, points[i - 1], points[i], color, thickness)

        return frame

    def draw_gesture_info(self, frame):
        """Draw gesture detection info on frame."""
        h, w = frame.shape[:2]

        # Gesture status panel
        cv2.rectangle(frame, (10, 90), (350, 200), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 90), (350, 200), (0, 255, 0), 2)

        y_offset = 110
        cv2.putText(
            frame,
            "GESTURE STATUS",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        y_offset += 30
        gesture_text = "Pointing: " + (
            "YES" if self.gesture_hand_position else "NO"
        )
        color = (0, 255, 0) if self.gesture_hand_position else (0, 0, 255)
        cv2.putText(frame, gesture_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        y_offset += 25
        open_hand_text = "Open Hand: " + (
            "YES" if self.control_hand_position else "NO"
        )
        color = (0, 255, 0) if self.control_hand_position else (0, 0, 255)
        cv2.putText(
            frame, open_hand_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )

        y_offset += 25
        cube_rotation_text = f"Cube: ({self.cube_rotation[0]:.1f}, {self.cube_rotation[1]:.1f})"
        cv2.putText(
            frame,
            cube_rotation_text,
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 0),
            1,
        )

        return frame

    def draw_enhanced_cube(self, frame, position, rotation, size=50):
        """Draw cube with additional visual effects."""
        # Draw glow effect
        cv2.circle(frame, position, size + 20, (50, 100, 150), 2)
        cv2.circle(frame, position, size + 30, (30, 60, 100), 1)

        # Draw regular cube
        self.draw_3d_cube(frame, position, rotation, size)

        # Draw rotation indicators
        angle_y = rotation[1] * 180 / np.pi
        angle_x = rotation[0] * 180 / np.pi

        indicator_len = int(size * 1.5)
        end_x = int(position[0] + indicator_len * np.cos(angle_y))
        end_y = int(position[1] + indicator_len * np.sin(angle_x))

        cv2.arrowedLine(frame, position, (end_x, end_y), (0, 255, 255), 2, tipLength=0.3)

        return frame

    def run_advanced(self, video_source=0, show_metrics=False):
        """Run advanced gesture stream with effects."""
        cap = cv2.VideoCapture(video_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Setup video writer if recording
        writer = None
        if self.enable_recording:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_path = os.path.join(self.output_dir, "gesture_stream.mp4")
            writer = cv2.VideoWriter(output_path, fourcc, 30.0, (1280, 720))

        print("Advanced Gesture Stream Active!")
        print("- Point with one hand to control the cube")
        print("- Open your other hand to switch images")
        print("- 'r' to toggle recording")
        print("- 'm' to toggle metrics display")
        print("- 'q' to quit")

        frame_count = 0
        recording_active = False
        last_toggle_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror for selfie view
            frame_timestamp_ms = int(time_module.time() * 1000) % (2**31)

            # Process frame
            frame = self.process_frame(frame, frame_timestamp_ms)

            # Draw trail
            if self.gesture_history:
                frame = self.draw_gesture_trail(frame)

            # Draw enhanced cube
            if self.gesture_hand_position is not None:
                frame = self.draw_enhanced_cube(
                    frame,
                    (int(self.gesture_hand_position[0]), int(self.gesture_hand_position[1])),
                    self.cube_rotation,
                    size=40,
                )
                self.gesture_history.append(self.gesture_hand_position)
                if len(self.gesture_history) > 100:
                    self.gesture_history.pop(0)

            # Overlay image
            if self.control_hand_position is not None:
                frame = self.overlay_image_on_hand(
                    frame, self.current_image, self.control_hand_position, scale=0.3
                )

            # Draw info
            frame = self.draw_gesture_info(frame)

            # Recording indicator
            if recording_active:
                cv2.putText(
                    frame,
                    "● RECORDING",
                    (1000, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                )

            frame_count += 1

            # Write frame if recording
            if recording_active and writer:
                writer.write(frame)

            cv2.imshow("Advanced Gesture Stream", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                recording_active = not recording_active
                print(f"Recording: {'ON' if recording_active else 'OFF'}")
            elif key == ord("m"):
                show_metrics = not show_metrics
                print(f"Metrics: {'ON' if show_metrics else 'OFF'}")

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        if self.enable_recording and writer:
            print(f"Video saved to: {self.output_dir}")


def main():
    """Run the advanced gesture stream."""
    stream = AdvancedGestureStream(
        image_path_1=None, image_path_2=None, enable_recording=False
    )
    stream.run_advanced(show_metrics=False)


if __name__ == "__main__":
    main()
