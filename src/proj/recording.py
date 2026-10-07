"""Interactive webcam recording with a traffic-light countdown."""

import csv
import time
from contextlib import ExitStack
from pathlib import Path

import cv2

from .preview import draw_landmarks, draw_status
from .sound import play_countdown_sound
from .tracking import (
    DEFAULT_MODEL_PATH,
    create_landmarker,
    detect_hands,
    write_hands,
    write_header,
)


def record(output_path="../../data/hand_data.csv", *, camera_index=0, countdown_seconds=3,
           sound_enabled=True, model_path=DEFAULT_MODEL_PATH):
    """Preview the camera, start with Space, and stop with Q or window close.

    M toggles sound. CSV timestamps and frame numbers begin on green.
    The output CSV is overwritten on each run. Returns its Path.
    """
    if not isinstance(countdown_seconds, int) or countdown_seconds < 1:
        raise ValueError("countdown_seconds must be a positive integer")
    output_path = Path(output_path)
    window_name = "Hand Tracking"
    frame_number = 0
    countdown_start = start_time = last_count = None
    last_timestamp = -1

    with ExitStack() as resources:
        resources.callback(cv2.destroyAllWindows)
        cap = cv2.VideoCapture(camera_index)
        resources.callback(cap.release)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")
        landmarker = create_landmarker(model_path)
        resources.callback(landmarker.close)
        csv_file = resources.enter_context(output_path.open("w", newline=""))
        writer = csv.writer(csv_file)
        write_header(writer)

        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            now = time.perf_counter()

            if countdown_start is None:
                draw_status(frame, "Press SPACE to start | Sound: " +
                            ("on" if sound_enabled else "off"), None)
            elif start_time is None:
                remaining = max(0, countdown_seconds - int(now - countdown_start))
                if remaining != last_count:
                    play_countdown_sound(go=remaining == 0, enabled=sound_enabled)
                    last_count = remaining
                if remaining > 0:
                    draw_status(frame, f"Get ready... {remaining}",
                                0 if remaining > 1 else 1)
                else:
                    start_time = now

            if start_time is not None:
                frame_number += 1
                # MediaPipe requires strictly increasing video timestamps.
                timestamp_ms = max(last_timestamp + 1, int((now - start_time) * 1000))
                last_timestamp = timestamp_ms
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hands = detect_hands(landmarker, rgb, timestamp_ms)
                write_hands(writer, hands, timestamp_ms, frame_number)
                draw_landmarks(frame, hands)
                draw_status(frame, "GO! Recording" if now - start_time < 1 else
                            f"Recording | {now - start_time:.1f}s", 2)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("m"):
                sound_enabled = not sound_enabled
            if key == ord(" ") and countdown_start is None:
                countdown_start = time.perf_counter()

    print(f"Saved to {output_path}")
    return output_path
