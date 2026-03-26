"""Mode 1 — Capture and display the first frame of the source video."""

import supervision as sv

from football_vision.config import AppConfig
from football_vision.core.video import get_frame_generator


def run(cfg: AppConfig, models: dict) -> None:
    frame_gen = get_frame_generator(cfg.source_video_path)
    frame = next(frame_gen)
    sv.plot_image(frame)
