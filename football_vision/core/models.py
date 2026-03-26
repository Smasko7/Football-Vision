"""Lazy model loaders — import and call only what you need."""

from inference import get_model
from ultralytics import YOLO


def load_player_model(model_id: str, api_key: str):
    """Load the Roboflow player-detection model."""
    return get_model(model_id=model_id, api_key=api_key)


def load_field_model(model_id: str, api_key: str):
    """Load the Roboflow field-keypoint detection model."""
    return get_model(model_id=model_id, api_key=api_key)


def load_yolo_model(path: str) -> YOLO:
    """Load a local YOLO model from *path*."""
    return YOLO(path)
