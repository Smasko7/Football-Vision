"""Mode 8 — Detect and visualise soccer pitch keypoints on a single frame."""

import supervision as sv

from football_vision.config import AppConfig
from football_vision.core.annotators import make_vertex_annotator
from football_vision.core.detection import detect_keypoints, filter_keypoints
from football_vision.core.video import get_frame_generator


def run(cfg: AppConfig, models: dict) -> None:
    frame_gen = get_frame_generator(cfg.source_video_path, start=200)
    frame = next(frame_gen)

    vertex_ann = make_vertex_annotator(cfg)

    key_points = detect_keypoints(models["field"], frame, cfg.detection_confidence)
    frame_reference_points, _ = filter_keypoints(key_points, cfg.keypoint_confidence_threshold)

    filtered_kp = sv.KeyPoints(xy=frame_reference_points[None, ...])

    annotated_frame = frame.copy()
    annotated_frame = vertex_ann.annotate(scene=annotated_frame, key_points=filtered_kp)

    sv.plot_image(annotated_frame)
