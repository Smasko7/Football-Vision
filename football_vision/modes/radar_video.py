"""Mode 10 — Full pipeline as a video: team classification + 2D pitch radar per frame."""

import numpy as np
import supervision as sv

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

from football_vision.config import AppConfig
from football_vision.core.detection import (
    detect_keypoints,
    detect_players,
    filter_keypoints,
    resolve_goalkeepers_team_id,
    split_detections,
)
from football_vision.core.video import get_frame_generator, get_video_info, make_video_sink


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def run(cfg: AppConfig, models: dict) -> None:
    pitch_config = SoccerPitchConfiguration()
    device = "cuda" if _cuda_available() else "cpu"
    classifier = TeamClassifier(device=device)

    tracker = sv.ByteTrack()
    tracker.reset()

    frame_gen = get_frame_generator(cfg.source_video_path)
    video_info = get_video_info(cfg.source_video_path)

    # Output video uses pitch dimensions, not source video dimensions
    sample_pitch = draw_pitch(pitch_config)
    h, w = sample_pitch.shape[:2]
    pitch_video_info = sv.VideoInfo(fps=video_info.fps, width=w, height=h)

    with make_video_sink(cfg.target_video_path, pitch_video_info) as sink:
        for i, frame in enumerate(frame_gen):
            print(f"Frame {i}")
            if i == cfg.radar_frame_limit:
                break

            # ── Detection ─────────────────────────────────────────────────
            detections = detect_players(models["player"], frame, cfg.detection_confidence)
            ball_det, other_det = split_detections(
                detections, cfg.ball_id, cfg.ball_pad_px, cfg.nms_threshold
            )
            other_det = tracker.update_with_detections(detections=other_det)

            goalkeepers_det = other_det[other_det.class_id == cfg.goalkeeper_id - 1]
            players_det = other_det[other_det.class_id == cfg.player_id - 1]
            referees_det = other_det[other_det.class_id == cfg.referee_id - 1]

            # ── Team classification ────────────────────────────────────────
            players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_det.xyxy]
            if players_crops:
                players_det.class_id = classifier.predict(players_crops)

            goalkeepers_det.class_id = resolve_goalkeepers_team_id(
                players_det, goalkeepers_det
            )
            referees_det.class_id -= 1

            players_for_pitch = sv.Detections.merge([players_det, goalkeepers_det])

            # ── Pitch projection ───────────────────────────────────────────
            key_points = detect_keypoints(models["field"], frame, cfg.detection_confidence)
            frame_pts, mask = filter_keypoints(key_points, cfg.keypoint_confidence_threshold)
            pitch_pts = np.array(pitch_config.vertices)[mask]

            transformer = ViewTransformer(source=frame_pts, target=pitch_pts)

            pitch_ball_xy = transformer.transform_points(
                ball_det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            )
            pitch_players_xy = transformer.transform_points(
                players_for_pitch.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            )
            pitch_referees_xy = transformer.transform_points(
                referees_det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            )

            # ── Render radar frame ─────────────────────────────────────────
            radar = draw_pitch(pitch_config)
            radar = draw_points_on_pitch(
                config=pitch_config, xy=pitch_ball_xy,
                face_color=sv.Color.WHITE, edge_color=sv.Color.BLACK, radius=10, pitch=radar,
            )
            radar = draw_points_on_pitch(
                config=pitch_config,
                xy=pitch_players_xy[players_for_pitch.class_id == 0],
                face_color=sv.Color.from_hex(f"#{cfg.color_team_0}"),
                edge_color=sv.Color.BLACK, radius=16, pitch=radar,
            )
            radar = draw_points_on_pitch(
                config=pitch_config,
                xy=pitch_players_xy[players_for_pitch.class_id == 1],
                face_color=sv.Color.from_hex(f"#{cfg.color_team_1}"),
                edge_color=sv.Color.BLACK, radius=16, pitch=radar,
            )
            radar = draw_points_on_pitch(
                config=pitch_config, xy=pitch_referees_xy,
                face_color=sv.Color.from_hex(f"#{cfg.color_referee}"),
                edge_color=sv.Color.BLACK, radius=16, pitch=radar,
            )

            sink.write_frame(radar)
