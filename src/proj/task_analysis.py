"""Key insertion timing and support for previous four-phase annotations."""

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .motor_metrics import movement_series
from .review_data import load_metadata, load_trial
from .trial_phases import EVENTS, KEY_EVENT

ANNOTATION_FILE = "task_annotations.json"
TASK_OUTCOMES = ["time_to_key_seconds", "post_key_duration_seconds", "recording_duration_seconds"]
LEGACY_OUTCOMES = ["task_duration_seconds", "unlock_duration_seconds",
                   "initial_rest_tip_dispersion_rms", "final_rest_tip_dispersion_rms"]


def load_annotations(csv_path):
    path = Path(csv_path).with_name(ANNOTATION_FILE)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def validate_annotations(annotation, duration_seconds):
    if KEY_EVENT in annotation:
        time = float(annotation[KEY_EVENT])
        if not np.isfinite(duration_seconds) or not np.isfinite(time) or time < 0 or time > duration_seconds:
            raise ValueError("Key-in-lock time must lie within the recording")
        return time
    times = np.array([annotation[event] for event in EVENTS], dtype=float)
    if not np.isfinite(times).all() or times[0] < 0 or times[-1] > duration_seconds:
        raise ValueError("Phase times must be finite and lie within the recording")
    if not (np.diff(times) > 0).all():
        raise ValueError("Use pickup start < unlocking start < unlocking end < return to rest")
    if annotation.get("outcome") not in ("success", "failed", "unknown"):
        raise ValueError("Outcome must be success, failed, or unknown")
    return times


def save_annotations(csv_path, annotation, duration_seconds):
    if KEY_EVENT in annotation:
        annotation = dict(annotation, outcome="success", success_assumed=True, source="review")
    validate_annotations(annotation, duration_seconds)
    result = dict(annotation, task="lock_and_key", annotated_at=datetime.now(timezone.utc).isoformat())
    Path(csv_path).with_name(ANNOTATION_FILE).write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def key_event_seconds(csv_path, metadata=None):
    annotation = load_annotations(csv_path) or {}
    value = annotation.get(KEY_EVENT, (metadata or {}).get(KEY_EVENT))
    return float(value) if value is not None else None


def add_task_metrics(trial_metrics, *, hand_selections=None, allow_legacy=False,
                     legacy_image_size=None, min_confidence=0.6,
                     max_gap_seconds=0.25, min_phase_frames=10):
    """Add key insertion times; preserve legacy phase analysis where available."""
    result = trial_metrics.copy()
    numeric = ["time_to_key_seconds", "post_key_duration_seconds", "key_in_lock_seconds",
               *LEGACY_OUTCOMES, "pickup_duration_seconds", "return_duration_seconds",
               "initial_rest_duration_seconds", "final_rest_duration_seconds",
               "initial_rest_frames", "final_rest_frames", "task_success"]
    for column in numeric:
        result[column] = np.nan
    result["task_outcome"] = "unknown"
    result["annotation_status"] = "not annotated"
    result["phase_quality_note"] = ""
    for index, row in result.iterrows():
        if not row.csv_path:
            continue
        try:
            annotation = load_annotations(row.csv_path)
            if annotation is None:
                continue
            if KEY_EVENT in annotation:
                metadata = load_metadata(row)
                duration = float(metadata["duration_seconds"])
                insertion = validate_annotations(annotation, duration)
                result.loc[index, "annotation_status"] = "annotated"
                result.loc[index, "task_outcome"] = "success"
                result.loc[index, "task_success"] = 1
                result.loc[index, "key_in_lock_seconds"] = insertion
                result.loc[index, "time_to_key_seconds"] = insertion
                result.loc[index, "post_key_duration_seconds"] = duration - insertion
                continue
            if any(event not in annotation for event in EVENTS):
                result.loc[index, "annotation_status"] = "key insertion not marked" if annotation.get("success_assumed") else "incomplete phase log"
                result.loc[index, "task_outcome"] = annotation.get("outcome", "unknown")
                continue
            metadata = load_metadata(row)
            duration = float(metadata["duration_seconds"])
            pickup, unlock_start, unlock_end, rest_return = validate_annotations(annotation, duration)
            result.loc[index, "annotation_status"] = "annotated"
            result.loc[index, "task_outcome"] = annotation["outcome"]
            if annotation["outcome"] != "unknown":
                result.loc[index, "task_success"] = int(annotation["outcome"] == "success")
            result.loc[index, "task_duration_seconds"] = rest_return - pickup
            result.loc[index, "pickup_duration_seconds"] = unlock_start - pickup
            result.loc[index, "unlock_duration_seconds"] = unlock_end - unlock_start
            result.loc[index, "return_duration_seconds"] = rest_return - unlock_end
            result.loc[index, "initial_rest_duration_seconds"] = pickup
            result.loc[index, "final_rest_duration_seconds"] = duration - rest_return
            # Rest motion still needs reliable selected-hand landmark data.
            if not row.analysis_eligible:
                result.loc[index, "phase_quality_note"] = "Rest features excluded by acquisition QC"
                continue
            series, _ = movement_series(
                load_trial(row.csv_path), metadata, (hand_selections or {}).get(row.trial_id),
                allow_legacy=allow_legacy, legacy_image_size=legacy_image_size,
                min_confidence=min_confidence, max_gap_seconds=max_gap_seconds,
            )
            notes = []
            for phase, start, end in (("initial_rest", 0, pickup), ("final_rest", rest_return, duration)):
                sample = series.loc[(series.time_seconds >= start) & (series.time_seconds < end)]
                result.loc[index, phase + "_frames"] = len(sample)
                if len(sample) < min_phase_frames:
                    notes.append(f"{phase}: fewer than {min_phase_frames} frames")
                    continue
                gaps = np.r_[sample.time_seconds.iloc[0] - start,
                             np.diff(sample.time_seconds), end - sample.time_seconds.iloc[-1]]
                if gaps.max() > max_gap_seconds:
                    notes.append(f"{phase}: detection gap exceeds {max_gap_seconds:g}s")
                    continue
                tip = sample[["relative_tip_x", "relative_tip_y"]].to_numpy()
                result.loc[index, phase + "_tip_dispersion_rms"] = np.sqrt(np.mean(np.sum((tip - tip.mean(axis=0)) ** 2, axis=1)))
            result.loc[index, "phase_quality_note"] = "; ".join(notes)
        except (ValueError, OSError, KeyError, TypeError) as error:
            result.loc[index, "annotation_status"] = f"Invalid annotation: {error}"
    return result


def outcome_counts(trials):
    """Counts are descriptive trial outcomes, not independent-subject tests."""
    if trials.empty:
        return trials[["group", "experiment_type", "task_outcome"]].assign(trials=[])
    return (trials.loc[trials.status == "completed"]
            .groupby(["group", "experiment_type", "task_outcome"]).size()
            .rename("trials").reset_index())
