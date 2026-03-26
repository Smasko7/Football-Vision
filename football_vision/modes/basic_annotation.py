"""Mode 2 — Annotate a single frame with bounding boxes and confidence labels."""

import supervision as sv

from football_vision.config import AppConfig
from football_vision.core.annotators import make_box_annotator, make_label_annotator
from football_vision.core.detection import detect_players
from football_vision.core.video import get_frame_generator


def run(cfg: AppConfig, models: dict) -> None:
    frame_gen = get_frame_generator(cfg.source_video_path)
    frame = next(frame_gen)

    box_annotator = make_box_annotator(cfg)
    label_annotator = make_label_annotator(cfg)

    detections = detect_players(
        models["player"], frame, cfg.detection_confidence
    )

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(detections["class_name"], detections.confidence)
    ]

    annotated_frame = frame.copy()
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels
    )

    sv.plot_image(annotated_frame)
