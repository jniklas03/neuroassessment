"""Single key-in-lock event logging on the recording clock."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path

# Retained for reading previous four-phase recordings.
EVENTS = ("pickup_start_seconds", "unlock_start_seconds", "unlock_end_seconds", "return_end_seconds")
KEY_EVENT = "key_in_lock_seconds"


@dataclass
class PhaseLog:
    events: dict = field(default_factory=dict)
    outcome: str = "success"

    @property
    def complete(self):
        return KEY_EVENT in self.events

    @property
    def phase(self):
        return "after key insertion" if self.complete else "before key insertion"

    @property
    def next_action(self):
        return "finish trial / next" if self.complete else "key is in the lock"

    def advance(self, seconds):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Event time must be finite and nonnegative")
        if self.complete:
            return True
        self.events[KEY_EVENT] = seconds
        return False

    def annotation(self, duration):
        return dict(self.events, task="lock_and_key", source="spacebar",
                    phase_log_complete=self.complete, outcome="success", success_assumed=True,
                    recording_duration_seconds=duration,
                    annotated_at=datetime.now(timezone.utc).isoformat(),
                    notes="Space marks key insertion. Success is assumed by protocol.")

    def metadata_events(self):
        return [dict(event="key_in_lock", timestamp_seconds=self.events[KEY_EVENT])] if self.complete else []

    def save(self, csv_path, duration):
        Path(csv_path).with_name("task_annotations.json").write_text(
            json.dumps(self.annotation(duration), indent=2) + "\n", encoding="utf-8"
        )
