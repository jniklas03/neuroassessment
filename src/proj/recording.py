"""Interactive webcam experiments with a traffic-light countdown."""

import csv
import time
from contextlib import ExitStack
from datetime import datetime, timezone

import cv2

from .preview import draw_landmarks, draw_status
from .participants import get_subject_group, get_dominant_hand
from .trial_phases import PhaseLog
from .session import (
    DEFAULT_DATA_DIR, create_session, get_subject_name, remaining_experiments, save_metadata,
)
from .sound import play_countdown_sound
from .tracking import (
    DEFAULT_MODEL_PATH, create_landmarker, detect_hands, write_hands, write_header,
)


def record(*, subject_name=None, group=None, dominant_hand=None, data_dir=DEFAULT_DATA_DIR, camera_index=0,
           countdown_seconds=3, sound_enabled=True, model_path=DEFAULT_MODEL_PATH):
    """Run the subject's unfinished experiments in random order in one window.

    Space starts the countdown, marks key insertion, then finishes the trial
    and loads the next experiment. Success is assumed. Q saves and exits.
    Trials without the key insertion marker remain unfinished.
    M toggles sound. Each experiment has its own CSV and metadata, with CSV
    timestamps starting at zero. Returns a list of saved CSV Paths.
    """
    if not isinstance(countdown_seconds, int) or countdown_seconds < 1:
        raise ValueError("countdown_seconds must be a positive integer")
    subject_name = get_subject_name(subject_name)
    output_paths = []
    if not remaining_experiments(subject_name, data_dir):
        print(f"All four experiments are complete for {subject_name}.")
        return output_paths
    group = get_subject_group(subject_name, data_dir, group)
    dominant_hand = get_dominant_hand(subject_name, data_dir, dominant_hand)
    window_name = "Hand Tracking"
    exit_requested = False
    # Keep MediaPipe timestamps increasing even when CSV timestamps reset.
    tracker_origin = time.perf_counter()
    last_tracker_timestamp = -1
    last_space_key_time = float("-inf")

    with ExitStack() as resources:
        resources.callback(cv2.destroyAllWindows)
        cap = cv2.VideoCapture(camera_index)
        resources.callback(cap.release)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")
        landmarker = create_landmarker(model_path)
        resources.callback(landmarker.close)

        while not exit_requested and remaining_experiments(subject_name, data_dir):
            directory, metadata = create_session(subject_name, data_dir)
            experiment_type = metadata["experiment_type"]
            print(f"Subject: {subject_name} | Experiment: {experiment_type}")
            output_path = directory / "hand_data.csv"
            frame_number = 0
            countdown_start = start_time = last_count = None
            last_timestamp = -1
            completed = False
            phases = PhaseLog()
            now = time.perf_counter()
            metadata.update(group=group, dominant_hand=dominant_hand,
                            task_hand=("left" if dominant_hand == "right" else "right")
                            if experiment_type.endswith("_ndom") else dominant_hand,
                            camera_index=camera_index, countdown_seconds=countdown_seconds,
                            sound_enabled=sound_enabled, frames_recorded=0,
                            duration_seconds=0.0, coordinate_system="mediapipe_normalized_image",
                            schema_version=2, image_mirrored=True, task_protocol="lock_and_key",
                            phase_logging="spacebar",
                            success_assumed=True,
                            task_description="Start at rest, shoulder width; pick up lock and key; unlock; return to rest")
            save_metadata(directory, metadata)

            try:
                with output_path.open("w", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    write_header(writer)
                    while True:
                        success, frame = cap.read()
                        if not success:
                            exit_requested = True
                            break
                        frame = cv2.flip(frame, 1)
                        metadata.update(image_height=frame.shape[0], image_width=frame.shape[1])
                        now = time.perf_counter()

                        if countdown_start is None:
                            draw_status(frame, f"{experiment_type} | SPACE: start | Sound: " +
                                        ("on" if sound_enabled else "off"), None)
                        elif start_time is None:
                            remaining = max(0, countdown_seconds - int(now - countdown_start))
                            if remaining != last_count:
                                play_countdown_sound(go=remaining == 0, enabled=sound_enabled)
                                last_count = remaining
                            if remaining > 0:
                                draw_status(frame, f"{experiment_type} | Get ready... {remaining}",
                                            0 if remaining > 1 else 1)
                            else:
                                start_time = now
                                metadata.update(
                                    status="recording",
                                    recording_started_at=datetime.now(timezone.utc).isoformat(),
                                )
                                save_metadata(directory, metadata)

                        if start_time is not None:
                            timestamp_ms = max(last_timestamp + 1, int((now - start_time) * 1000))
                            last_timestamp = timestamp_ms
                            tracker_timestamp = max(
                                last_tracker_timestamp + 1, int((now - tracker_origin) * 1000)
                            )
                            last_tracker_timestamp = tracker_timestamp
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            hands, handedness = detect_hands(
                                landmarker, rgb, tracker_timestamp, include_handedness=True
                            )
                            frame_number += 1
                            write_hands(writer, hands, timestamp_ms, frame_number, handedness)
                            draw_landmarks(frame, hands)
                            draw_status(frame, f"{experiment_type} | {phases.phase} | {now - start_time:.1f}s", 2,
                                        instructions=f"SPACE: {phases.next_action} | Q: exit | M: sound")

                        cv2.imshow(window_name, frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q") or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                            completed = start_time is not None and phases.complete and frame_number > 0
                            exit_requested = True
                            break
                        if key == ord("m"):
                            sound_enabled = not sound_enabled
                        space_press = key == ord(" ") and now - last_space_key_time >= 0.3
                        if key == ord(" "):
                            # Ignore OS key-repeat while Space is held down.
                            last_space_key_time = now
                        if space_press:
                            if start_time is not None:
                                completed = phases.advance(now - start_time)
                                phases.save(output_path, now - start_time)
                                metadata.update(phase_events=phases.metadata_events(),
                                                phase_log_complete=phases.complete,
                                                task_outcome=phases.outcome)
                                metadata.update(phases.events)
                                csv_file.flush()
                                save_metadata(directory, metadata)
                                if completed:
                                    break
                            if countdown_start is None:
                                countdown_start = now
            finally:
                metadata.update(
                    status="completed" if completed else (
                        "interrupted" if start_time is not None else "cancelled"
                    ),
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    frames_recorded=frame_number,
                    duration_seconds=max(0.0, now - start_time) if start_time is not None else 0.0,
                    sound_enabled=sound_enabled,
                    phase_events=phases.metadata_events() if start_time is not None else [],
                    phase_log_complete=phases.complete,
                    task_outcome=phases.outcome,
                )
                if start_time is not None:
                    phases.save(output_path, metadata["duration_seconds"])
                save_metadata(directory, metadata)
            output_paths.append(output_path)
            print(f"Saved to {output_path}")

        if not exit_requested:
            print(f"All four experiments are complete for {subject_name}.")
            while True:
                success, frame = cap.read()
                if not success:
                    break
                frame = cv2.flip(frame, 1)
                draw_status(frame, "All four experiments complete! Q: exit", 2)
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

    return output_paths
