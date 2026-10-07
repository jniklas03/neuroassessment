"""Optional, nonblocking countdown audio cues."""

import io
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import threading
import wave


def play_countdown_sound(go=False, enabled=True):
    """Play a short cue in the background when an audio player is available."""
    if not enabled:
        return

    def play():
        frequency = 1000 if go else 600
        duration = 0.25 if go else 0.12
        try:
            if sys.platform == "win32":
                import winsound
                winsound.Beep(frequency, int(duration * 1000))
                return
            player = shutil.which("aplay") or shutil.which("paplay")
            if player is None:
                return  # The visual countdown works without sound support.
            sample_rate = 22050
            samples = b"".join(
                struct.pack("<h", int(8000 * math.sin(2 * math.pi * frequency * i / sample_rate)))
                for i in range(int(sample_rate * duration))
            )
            audio = io.BytesIO()
            with wave.open(audio, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(samples)
            args = [player, "-q"] if Path(player).name == "aplay" else [player, "/dev/stdin"]
            subprocess.run(args, input=audio.getvalue(), stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=2)
        except (OSError, subprocess.SubprocessError):
            pass

    threading.Thread(target=play, daemon=True).start()


