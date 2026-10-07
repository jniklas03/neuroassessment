"""Camera preview overlays."""

import cv2


def draw_status(frame, message, active_light, *, instructions=None):
    """Draw a traffic light and readable status over the camera preview."""
    height, width = frame.shape[:2]
    panel_height = min(110, height)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, panel_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    colors = [(0, 0, 255), (0, 190, 255), (0, 220, 0)]
    for index, color in enumerate(colors):
        cv2.circle(frame, (30 + index * 48, 30), 16,
                   color if index == active_light else (65, 65, 65), -1)
    scale = min(0.65, max(0.3, (width - 24) / 620))
    cv2.putText(frame, message, (12, 72), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 2, cv2.LINE_AA)
    instructions = instructions or "SPACE: countdown | Q: exit | M: sound"
    instruction_scale = min(scale * 0.8, (width - 24) / max(1, cv2.getTextSize(
        instructions, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0][0]))
    cv2.putText(frame, instructions, (12, 98),
                cv2.FONT_HERSHEY_SIMPLEX, instruction_scale,
                (210, 210, 210), 1, cv2.LINE_AA)


def draw_landmarks(frame, hands):
    """Draw each detected hand's landmarks on the preview."""
    height, width = frame.shape[:2]
    for hand in hands:
        for landmark in hand:
            cv2.circle(frame, (int(landmark.x * width), int(landmark.y * height)),
                       4, (0, 255, 0), -1)
