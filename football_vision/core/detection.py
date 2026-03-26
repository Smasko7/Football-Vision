"""Shared detection helpers used across multiple modes."""

from typing import Tuple

import numpy as np
import supervision as sv
from ultralytics import YOLO


def detect_players(model, frame, confidence: float) -> sv.Detections:
    """Run the player-detection model on *frame* and return raw detections.

    Supports both a locally trained YOLO model (ultralytics) and a
    Roboflow inference API model — the correct path is chosen automatically.
    """
    if isinstance(model, YOLO):
        result = model(frame, conf=confidence)[0]
        return sv.Detections.from_ultralytics(result)
    else:
        result = model.infer(frame, confidence=confidence)[0]
        return sv.Detections.from_inference(result)


def split_detections(
    detections: sv.Detections,
    ball_id: int,
    pad_px: int,
    nms_threshold: float,
) -> Tuple[sv.Detections, sv.Detections]:
    """Split raw detections into ball and all-other detections.

    Returns:
        ball_detections: padded bounding boxes for the ball
        other_detections: NMS-filtered non-ball detections with class_id
            decremented by 1 (so goalkeeper=0, player=1, referee=2)
    """
    ball_detections = detections[detections.class_id == ball_id]
    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=pad_px)

    other_detections = detections[detections.class_id != ball_id]
    other_detections = other_detections.with_nms(
        threshold=nms_threshold, class_agnostic=True
    )
    other_detections.class_id -= 1

    return ball_detections, other_detections


def detect_keypoints(model, frame, confidence: float) -> sv.KeyPoints:
    """Run the field-detection model on *frame* and return raw key points."""
    result = model.infer(frame, confidence=confidence)[0]
    return sv.KeyPoints.from_inference(result)


def filter_keypoints(
    key_points: sv.KeyPoints,
    confidence_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter key points by confidence.

    Returns:
        frame_reference_points: (N, 2) array of filtered xy coordinates
        confidence_mask: boolean mask used for filtering pitch vertices
    """
    mask = key_points.confidence[0] > confidence_threshold
    frame_reference_points = key_points.xy[0][mask]
    return frame_reference_points, mask


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    goalkeepers: sv.Detections,
) -> np.ndarray:
    """Assign each goalkeeper to the nearest team centroid.

    Args:
        players: detections with class_id already set to 0 or 1 (team labels)
        goalkeepers: goalkeeper detections to assign

    Returns:
        Array of team IDs (0 or 1) with one entry per goalkeeper.
    """
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players.class_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players.class_id == 1].mean(axis=0)

    goalkeepers_team_id = []
    for gk_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(gk_xy - team_0_centroid)
        dist_1 = np.linalg.norm(gk_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)

    return np.array(goalkeepers_team_id)
