"""Load trial inventory and validate landmark CSVs without camera dependencies."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .participants import GROUPS, read_groups, read_dominant_hands, write_groups
from .session import DEFAULT_DATA_DIR, EXPERIMENT_TYPES

COORDINATES = [f"{axis}{index}" for index in range(21) for axis in "xyz"]
INDEX_COLUMNS = [
    "trial_id", "subject_name", "group", "experiment_type", "status", "frames_recorded",
    "duration_seconds", "dominant_hand", "csv_path", "metadata_path", "error",
]


def discover_trials(data_dir=DEFAULT_DATA_DIR):
    """Include completed, cancelled, and broken trials in the inventory."""
    root = Path(data_dir)
    assignments = read_groups(root)
    dominant_hands = read_dominant_hands(root)
    rows = []
    for path in sorted(root.glob("*/*/metadata.json")):
        row = dict.fromkeys(INDEX_COLUMNS, "")
        row.update(trial_id=str(path.parent.relative_to(root)), metadata_path=str(path))
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            name = metadata["subject_name"]
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Invalid subject name")
            csv_name = metadata.get("data_file", "hand_data.csv")
            if Path(csv_name).name != csv_name:
                raise ValueError("data_file must be a filename within the trial")
            row.update(
                subject_name=name,
                group=assignments.get(name) or metadata.get("group", ""),
                dominant_hand=dominant_hands.get(name) or metadata.get("dominant_hand", ""),
                experiment_type=metadata.get("experiment_type", ""),
                status=metadata.get("status", "unknown"),
                frames_recorded=metadata.get("frames_recorded", np.nan),
                duration_seconds=metadata.get("duration_seconds", np.nan),
                csv_path=str(path.parent / csv_name),
            )
            if not Path(row["csv_path"]).is_file():
                raise ValueError("Landmark CSV is missing")
            if row["experiment_type"] not in EXPERIMENT_TYPES:
                raise ValueError("Unknown or missing experiment type")
            if row["group"] not in ("", *GROUPS):
                raise ValueError("Unknown group label")
        except (ValueError, OSError, KeyError, TypeError) as error:
            row["error"] = str(error)
        rows.append(row)
    result = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    # A participant must have a single group across all conditions.
    for subject, trials in result.groupby("subject_name"):
        labels = set(trials["group"]) - {""}
        if len(labels) > 1:
            result.loc[result.subject_name == subject, "error"] = "Conflicting group labels; fix subject_groups.csv"
    return result


def prepare_group_file(inventory, data_dir=DEFAULT_DATA_DIR):
    """Add missing subjects to the editable registry; preserve existing assignments."""
    assignments = read_groups(data_dir)
    for name, trials in inventory.loc[inventory.subject_name != ""].groupby("subject_name"):
        if name not in assignments:
            known = set(trials.group) & set(GROUPS)
            assignments[name] = next(iter(known)) if len(known) == 1 else ""
    write_groups(data_dir, assignments)
    return Path(data_dir) / "subject_groups.csv"


def load_trial(csv_path):
    """Require finite coordinates, unique hand/frame rows, and valid frame clocks."""
    data = pd.read_csv(csv_path)
    required = ["timestamp_ms", "frame", "hand", *COORDINATES]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if not np.isfinite(data[required].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite timestamps, indices, or landmark coordinates")
    for column in ("frame", "hand"):
        if ((data[column] < 0) | (data[column] % 1 != 0)).any():
            raise ValueError(f"{column} must contain nonnegative integers")
    if (data.timestamp_ms < 0).any():
        raise ValueError("Negative recording timestamps")
    if data.duplicated(["frame", "hand"]).any():
        raise ValueError("Duplicate hand rows within a frame")
    frame_times = data.groupby("frame").timestamp_ms
    if (frame_times.nunique() > 1).any():
        raise ValueError("Hands in the same frame have inconsistent timestamps")
    if (frame_times.first().sort_index().diff().dropna() <= 0).any():
        raise ValueError("Recording timestamps must increase with frame numbers")
    if "handedness" not in data:
        data["handedness"] = "unknown"
    data["handedness"] = data.handedness.fillna("unknown").astype("string").str.lower()
    if not data.handedness.isin(["left", "right", "unknown"]).all():
        raise ValueError("Unexpected anatomical hand label")
    if "handedness_score" not in data:
        data["handedness_score"] = np.nan
    data["handedness_score"] = pd.to_numeric(data.handedness_score, errors="coerce")
    scores = data.handedness_score.dropna()
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Handedness confidence must be between zero and one")
    return data.sort_values(["timestamp_ms", "hand"]).reset_index(drop=True)


def load_metadata(row):
    return json.loads(Path(row["metadata_path"]).read_text(encoding="utf-8"))
