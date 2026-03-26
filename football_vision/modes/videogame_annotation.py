"""Modes 3 & 4 — Video-game style annotations (single frame or full video)."""

import supervision as sv

from football_vision.config import AppConfig
from football_vision.core.annotators import make_ellipse_annotator, make_triangle_annotator
from football_vision.core.detection import detect_players, split_detections
from football_vision.core.video import get_frame_generator, get_video_info, make_video_sink


def _annotate_frame(frame, ball_detections, other_detections, ellipse_ann, triangle_ann):
    annotated = frame.copy()
    annotated = ellipse_ann.annotate(scene=annotated, detections=other_detections)
    annotated = triangle_ann.annotate(scene=annotated, detections=ball_detections)
    return annotated


def run_frame(cfg: AppConfig, models: dict) -> None:
    """Mode 3 — video-game style annotation on a single frame."""
    frame_gen = get_frame_generator(cfg.source_video_path)
    frame = next(frame_gen)

    ellipse_ann = make_ellipse_annotator(cfg)
    triangle_ann = make_triangle_annotator(cfg)

    detections = detect_players(models["player"], frame, cfg.detection_confidence)
    ball_det, other_det = split_detections(
        detections, cfg.ball_id, cfg.ball_pad_px, cfg.nms_threshold
    )

    annotated_frame = _annotate_frame(frame, ball_det, other_det, ellipse_ann, triangle_ann)
    sv.plot_image(annotated_frame)


def run_video(cfg: AppConfig, models: dict) -> None:
    """Mode 4 — video-game style annotation written to a video file."""
    ellipse_ann = make_ellipse_annotator(cfg)
    triangle_ann = make_triangle_annotator(cfg)

    video_info = get_video_info(cfg.source_video_path)
    frame_gen = get_frame_generator(cfg.source_video_path)

    with make_video_sink(cfg.target_video_path, video_info) as sink:
        for i, frame in enumerate(frame_gen):
            if i == cfg.annotation_frame_limit:
                break

            detections = detect_players(models["player"], frame, cfg.detection_confidence)
            ball_det, other_det = split_detections(
                detections, cfg.ball_id, cfg.ball_pad_px, cfg.nms_threshold
            )

            annotated_frame = _annotate_frame(
                frame, ball_det, other_det, ellipse_ann, triangle_ann
            )
            sink.write_frame(annotated_frame)


def run(cfg: AppConfig, models: dict, as_video: bool = False) -> None:
    if as_video:
        run_video(cfg, models)
    else:
        run_frame(cfg, models)
