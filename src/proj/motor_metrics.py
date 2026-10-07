"""Exploratory motor features and acquisition-quality checks."""

import numpy as np
import pandas as pd

from .review_data import COORDINATES, load_metadata, load_trial

METRICS = ["tip_dispersion_rms", "pinch_sd", "pinch_mean", "tip_speed_mean", "tip_path_length"]
PRIMARY_METRICS = ["tip_dispersion_rms", "pinch_sd"]


def select_hand(data, selection=None, *, allow_legacy=False, min_confidence=0.6):
    """Use anatomical labels; never treat a changing detection index as identity."""
    labels = set(data.handedness) & {"left", "right"}
    if selection is None:
        if len(labels) == 1:
            selection = next(iter(labels))
        elif not labels and allow_legacy and data.hand.nunique() == 1:
            selection = int(data.hand.iloc[0])
        else:
            raise ValueError("Select the task hand explicitly; legacy hand indices are not stable identities")
    if isinstance(selection, str) and selection.lower() in ("left", "right"):
        name = selection.lower()
        selected = data.loc[(data.handedness == name) & (data.handedness_score >= min_confidence)].copy()
    elif isinstance(selection, (int, np.integer)) and allow_legacy:
        name = f"legacy_index_{selection}"
        selected = data.loc[data.hand == selection].copy()
    else:
        raise ValueError("Select left/right, or enable allow_legacy before selecting a numeric index")
    # Duplicate anatomical labels in a frame are ambiguous; discard that frame.
    selected = selected.loc[~selected.duplicated("frame", keep=False)]
    if selected.empty:
        raise ValueError("No unambiguous detections of the selected hand")
    return selected.sort_values("timestamp_ms"), name


def movement_series(data, metadata, selection=None, *, allow_legacy=False,
                    legacy_image_size=None, min_confidence=0.6, max_gap_seconds=0.25):
    if selection is None:
        selection = metadata.get("task_hand")
    selected, name = select_hand(data, selection, allow_legacy=allow_legacy,
                                 min_confidence=min_confidence)
    width, height = metadata.get("image_width"), metadata.get("image_height")
    if not width or not height:
        if legacy_image_size is None:
            raise ValueError("Missing image dimensions: set legacy_image_size=(width, height) from the actual camera")
        width, height = legacy_image_size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    points = selected[COORDINATES].to_numpy().reshape(-1, 21, 3)
    xy = points[:, :, :2].copy()
    xy[:, :, 1] *= height / width  # Both axes in units of image width.
    palm_lengths = np.linalg.norm(xy[:, 9] - xy[:, 0], axis=1)
    scale = np.median(palm_lengths)
    if scale <= 1e-6:
        raise ValueError("Degenerate wrist-to-middle-MCP reference length")
    # A fixed per-trial scale avoids injecting frame-by-frame scale noise.
    relative_tip = (xy[:, 8] - xy[:, 0]) / scale
    pinch = np.linalg.norm(xy[:, 8] - xy[:, 4], axis=1) / scale
    times = selected.timestamp_ms.to_numpy(dtype=float) / 1000
    dt = np.diff(times)
    frame_step = np.diff(selected.frame.to_numpy())
    valid_edge = (dt > 0) & (dt <= max_gap_seconds) & (frame_step == 1)
    step_length = np.linalg.norm(np.diff(relative_tip, axis=0), axis=1)
    speeds = np.full(len(times), np.nan)
    speeds[1:][valid_edge] = step_length[valid_edge] / dt[valid_edge]
    return pd.DataFrame({
        "time_seconds": times, "frame": selected.frame.to_numpy(),
        "relative_tip_x": relative_tip[:, 0], "relative_tip_y": relative_tip[:, 1],
        "pinch": pinch, "tip_speed": speeds,
        "valid_step_length": np.r_[np.nan, np.where(valid_edge, step_length, np.nan)],
        "valid_step_duration": np.r_[np.nan, np.where(valid_edge, dt, np.nan)],
    }), name


def extract_metrics(inventory, *, hand_selections=None, allow_legacy=False,
                    legacy_image_size=None, min_frames=30, min_coverage=0.8,
                    min_duration_seconds=1.0, min_confidence=0.6, max_gap_seconds=0.25):
    """Return one row per trial, including explicit exclusion reasons."""
    rows = []
    hand_selections = hand_selections or {}
    for _, trial in inventory.iterrows():
        row = trial.to_dict()
        row["recording_duration_seconds"] = pd.to_numeric(trial.duration_seconds, errors="coerce")
        row.update({metric: np.nan for metric in METRICS})
        row.update(selected_hand="", selected_frames=0, detection_coverage=np.nan,
                   max_detection_gap_seconds=np.nan, analysis_eligible=False, exclusion_reason="")
        reasons = []
        if trial["error"]:
            reasons.append(trial["error"])
        if trial.status != "completed":
            reasons.append(f"Trial status: {trial.status}")
        try:
            if trial["error"]:
                raise ValueError(trial["error"])
            data = load_trial(trial.csv_path)
            metadata = load_metadata(trial)
            series, name = movement_series(
                data, metadata, hand_selections.get(trial.trial_id), allow_legacy=allow_legacy,
                legacy_image_size=legacy_image_size, min_confidence=min_confidence,
                max_gap_seconds=max_gap_seconds,
            )
            n = len(series)
            frames_recorded = metadata.get("frames_recorded")
            if not isinstance(frames_recorded, (int, float)) or frames_recorded <= 0:
                raise ValueError("Missing/invalid frames_recorded: coverage cannot be assessed")
            if n > frames_recorded or (len(data) and data.frame.max() > frames_recorded):
                raise ValueError("CSV frame counts exceed metadata")
            coverage = n / frames_recorded
            elapsed = float(series.time_seconds.iloc[-1] - series.time_seconds.iloc[0])
            observed_duration = max(elapsed, float(metadata.get("duration_seconds", elapsed)))
            gaps = np.diff(series.time_seconds)
            max_gap = max(float(gaps.max()) if len(gaps) else 0.0,
                          float(series.time_seconds.iloc[0]),
                          max(0.0, observed_duration - float(series.time_seconds.iloc[-1])))
            tip = series[["relative_tip_x", "relative_tip_y"]].to_numpy()
            row.update(
                selected_hand=name, selected_frames=n, detection_coverage=coverage,
                max_detection_gap_seconds=max_gap,
                tip_dispersion_rms=float(np.sqrt(np.mean(np.sum((tip - tip.mean(axis=0)) ** 2, axis=1)))),
                pinch_mean=float(series.pinch.mean()), pinch_sd=float(series.pinch.std(ddof=1)),
                tip_speed_mean=float(series.valid_step_length.sum(min_count=1) /
                                     series.valid_step_duration.sum(min_count=1)),
                tip_path_length=float(series.valid_step_length.sum(min_count=1)),
            )
            if n < min_frames:
                reasons.append(f"Fewer than {min_frames} selected frames")
            if coverage < min_coverage:
                reasons.append(f"Detection coverage below {min_coverage:.0%}")
            if elapsed < min_duration_seconds:
                reasons.append(f"Observed duration below {min_duration_seconds:g}s")
            if series.tip_speed.notna().sum() < 2:
                reasons.append("Insufficient consecutive frames for speed/path estimates")
        except (ValueError, OSError, KeyError, TypeError) as error:
            reasons.append(str(error))
        row["exclusion_reason"] = "; ".join(dict.fromkeys(reasons))
        row["analysis_eligible"] = not reasons
        rows.append(row)
    columns = [*inventory.columns, "recording_duration_seconds", *METRICS, "selected_hand", "selected_frames",
               "detection_coverage", "max_detection_gap_seconds", "analysis_eligible", "exclusion_reason"]
    return pd.DataFrame(rows, columns=columns)
