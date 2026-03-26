"""Mode 5 — Track individual players across frames using ByteTrack."""

import supervision as sv

from football_vision.config import AppConfig
from football_vision.core.annotators import (
    make_ellipse_annotator,
    make_label_annotator,
    make_triangle_annotator,
)
from football_vision.core.detection import detect_players, split_detections
from football_vision.core.video import get_frame_generator, get_video_info, make_video_sink


def run(cfg: AppConfig, models: dict) -> None:
    ellipse_ann = make_ellipse_annotator(cfg)
    label_ann = make_label_annotator(cfg)
    triangle_ann = make_triangle_annotator(cfg)

    video_info = get_video_info(cfg.source_video_path)
    frame_gen = get_frame_generator(cfg.source_video_path)

    tracker = sv.ByteTrack()
    tracker.reset()

    with make_video_sink(cfg.target_video_path, video_info) as sink:
        for i, frame in enumerate(frame_gen):
            if i == cfg.tracking_frame_limit:
                break

            detections = detect_players(models["player"], frame, cfg.detection_confidence)
            ball_det, other_det = split_detections(
                detections, cfg.ball_id, cfg.ball_pad_px, cfg.nms_threshold
            )
            other_det = tracker.update_with_detections(detections=other_det)

            labels = [f"#{tid}" for tid in other_det.tracker_id]

            annotated_frame = frame.copy()
            annotated_frame = ellipse_ann.annotate(scene=annotated_frame, detections=other_det)
            annotated_frame = label_ann.annotate(
                scene=annotated_frame, detections=other_det, labels=labels
            )
            annotated_frame = triangle_ann.annotate(
                scene=annotated_frame, detections=ball_det
            )

            sink.write_frame(annotated_frame)
