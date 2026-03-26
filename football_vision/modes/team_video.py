"""Mode 7 — Per-frame team classification using TeamClassifier (SigLIP + UMAP + KMeans)."""

import supervision as sv

from sports.common.team import TeamClassifier

from football_vision.config import AppConfig
from football_vision.core.annotators import (
    make_ellipse_annotator,
    make_label_annotator,
    make_triangle_annotator,
)
from football_vision.core.detection import detect_players, split_detections
from football_vision.core.video import get_frame_generator, get_video_info, make_video_sink


def run(cfg: AppConfig, models: dict) -> None:
    classifier = TeamClassifier(device="cuda" if _cuda_available() else "cpu")

    ellipse_ann = make_ellipse_annotator(cfg)
    label_ann = make_label_annotator(cfg)
    triangle_ann = make_triangle_annotator(cfg)

    video_info = get_video_info(cfg.source_video_path)
    frame_gen = get_frame_generator(cfg.source_video_path)

    tracker = sv.ByteTrack()
    tracker.reset()

    with make_video_sink(cfg.target_video_path, video_info) as sink:
        for i, frame in enumerate(frame_gen):
            print(f"Frame {i}")
            if i == cfg.team_video_frame_limit:
                break

            detections = detect_players(models["player"], frame, cfg.detection_confidence)
            ball_det, other_det = split_detections(
                detections, cfg.ball_id, cfg.ball_pad_px, cfg.nms_threshold
            )
            other_det = tracker.update_with_detections(detections=other_det)

            players_det = other_det[other_det.class_id == cfg.player_id - 1]
            players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_det.xyxy]

            if players_crops:
                team_ids = classifier.predict(players_crops)
                players_det.class_id = team_ids

            labels = [f"#{tid}" for tid in players_det.tracker_id]

            annotated_frame = frame.copy()
            annotated_frame = ellipse_ann.annotate(
                scene=annotated_frame, detections=players_det
            )
            annotated_frame = label_ann.annotate(
                scene=annotated_frame, detections=players_det, labels=labels
            )
            annotated_frame = triangle_ann.annotate(
                scene=annotated_frame, detections=ball_det
            )

            sink.write_frame(annotated_frame)


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
