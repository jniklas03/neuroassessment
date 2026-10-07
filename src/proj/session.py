"""Subject prompts and per-recording storage."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
EXPERIMENT_TYPES = ("normal_dom", "normal_ndom", "blind_dom", "blind_ndom")


def get_subject_name(subject_name=None):
    if subject_name is not None:
        name = subject_name.strip()
        if not name:
            raise ValueError("Subject name must not be empty")
        return name
    while True:
        name = input("Subject name (Ctrl+C to cancel): ").strip()
        if name:
            return name
        print("Please enter a subject name.")


def subject_directory(subject_name, data_dir=DEFAULT_DATA_DIR):
    """Use the subject's name directly, without a generated suffix."""
    if (
        not subject_name.strip()
        or subject_name in (".", "..")
        or any(character in subject_name for character in ('/', '\\', '\x00'))
    ):
        raise ValueError("Subject name must be a nonempty folder name without path separators")
    return Path(data_dir) / subject_name


def _legacy_subject_directory(subject_name, data_dir):
    """Locate recordings made with the previous hashed folder names."""
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", subject_name).strip("_")[:64] or "subject"
    # Distinguish names that produce the same filesystem-safe slug.
    digest = hashlib.sha256(subject_name.encode("utf-8")).hexdigest()[:10]
    return Path(data_dir) / f"{slug}_{digest}"


def remaining_experiments(subject_name, data_dir=DEFAULT_DATA_DIR):
    """Find unfinished experiments using this subject's saved metadata."""
    completed = set()
    directories = (
        subject_directory(subject_name, data_dir),
        _legacy_subject_directory(subject_name, data_dir),
    )
    for directory in directories:
        for path in directory.glob("*/metadata.json"):
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if metadata.get("subject_name") == subject_name and metadata.get("status") == "completed":
                completed.add(metadata.get("experiment_type"))
    return [experiment for experiment in EXPERIMENT_TYPES if experiment not in completed]


def create_session(subject_name, data_dir=DEFAULT_DATA_DIR):
    """Randomly select one of the subject's unfinished experiments."""
    remaining = remaining_experiments(subject_name, data_dir)
    if not remaining:
        raise ValueError(f"All four experiments are already completed for {subject_name}.")
    experiment_type = random.choice(remaining)
    created_at = datetime.now(timezone.utc).astimezone()
    session_id = f"{experiment_type}_{created_at:%Y-%m-%d_%H-%M-%S_%f}"
    directory = subject_directory(subject_name, data_dir) / session_id
    directory.mkdir(parents=True, exist_ok=False)
    metadata = {
        "subject_name": subject_name,
        "experiment_type": experiment_type,
        "session_id": session_id,
        "created_at": created_at.isoformat(),
        "recording_started_at": None,
        "data_file": "hand_data.csv",
        "status": "ready",
    }
    return directory, metadata


def save_metadata(directory, metadata):
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
