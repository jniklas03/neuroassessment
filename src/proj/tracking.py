"""MediaPipe hand tracker setup and CSV serialization."""

import urllib.request
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"


def create_landmarker(model_path=DEFAULT_MODEL_PATH):
    """Download a missing model and create a video-mode hand tracker."""
    model_path = Path(model_path)
    if not model_path.is_file():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, model_path)
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_hands(landmarker, frame, timestamp_ms):
    """Detect hands in an RGB frame at the given recording timestamp."""
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    return landmarker.detect_for_video(image, timestamp_ms).hand_landmarks


def write_header(writer):
    header = ["timestamp_ms", "frame", "hand"]
    for index in range(21):
        header.extend(f"{axis}{index}" for axis in ("x", "y", "z"))
    writer.writerow(header)


def write_hands(writer, hands, timestamp_ms, frame_number):
    for hand_index, hand in enumerate(hands):
        row = [timestamp_ms, frame_number, hand_index]
        for landmark in hand:
            row.extend((landmark.x, landmark.y, landmark.z))
        writer.writerow(row)
