"""Mode 9 — Full pipeline on a single frame: team classification + 2D pitch radar."""

import numpy as np
import supervision as sv

from sports.annotators.soccer import (
    draw_pitch,
    draw_pitch_voronoi_diagram,
    draw_points_on_pitch,
)
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

from football_vision.config import AppConfig
from football_vision.core.annotators import (
    make_ellipse_annotator,
    make_label_annotator,
    make_triangle_annotator,
)
from football_vision.core.detection import (
    detect_keypoints,
    detect_players,
    filter_keypoints,
    resolve_goalkeepers_team_id,
    split_detections,
)
from football_vision.core.video import get_frame_generator


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

    ellipse_ann = make_ellipse_annotator(cfg)
    label_ann = make_label_annotator(cfg)
    triangle_ann = make_triangle_annotator(cfg)

    tracker = sv.ByteTrack()
    tracker.reset()

    frame_gen = get_frame_generator(cfg.source_video_path)
    frame = next(frame_gen)

    # ── Detection ──────────────────────────────────────────────────────────
    detections = detect_players(models["player"], frame, cfg.detection_confidence)
    ball_det, other_det = split_detections(
        detections, cfg.ball_id, cfg.ball_pad_px, cfg.nms_threshold
    )
    other_det = tracker.update_with_detections(detections=other_det)

    goalkeepers_det = other_det[other_det.class_id == cfg.goalkeeper_id - 1]
    players_det = other_det[other_det.class_id == cfg.player_id - 1]
    referees_det = other_det[other_det.class_id == cfg.referee_id - 1]

    # ── Team classification ────────────────────────────────────────────────
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_det.xyxy]
    if players_crops:
        players_det.class_id = classifier.predict(players_crops)

    goalkeepers_det.class_id = resolve_goalkeepers_team_id(players_det, goalkeepers_det)
    referees_det.class_id -= 1

    all_det = sv.Detections.merge([players_det, goalkeepers_det, referees_det])
    all_det.class_id = all_det.class_id.astype(int)

    # ── Frame visualisation ────────────────────────────────────────────────
    labels = [f"#{tid}" for tid in all_det.tracker_id]

    annotated_frame = frame.copy()
    annotated_frame = ellipse_ann.annotate(scene=annotated_frame, detections=all_det)
    annotated_frame = label_ann.annotate(
        scene=annotated_frame, detections=all_det, labels=labels
    )
    annotated_frame = triangle_ann.annotate(scene=annotated_frame, detections=ball_det)
    sv.plot_image(annotated_frame)

    # ── Pitch projection ───────────────────────────────────────────────────
    players_for_pitch = sv.Detections.merge([players_det, goalkeepers_det])

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

    # ── Radar view ─────────────────────────────────────────────────────────
    radar = draw_pitch(pitch_config)
    radar = draw_points_on_pitch(
        config=pitch_config, xy=pitch_ball_xy,
        face_color=sv.Color.WHITE, edge_color=sv.Color.BLACK, radius=10, pitch=radar,
    )
    radar = draw_points_on_pitch(
        config=pitch_config, xy=pitch_players_xy[players_for_pitch.class_id == 0],
        face_color=sv.Color.from_hex(f"#{cfg.color_team_0}"),
        edge_color=sv.Color.BLACK, radius=16, pitch=radar,
    )
    radar = draw_points_on_pitch(
        config=pitch_config, xy=pitch_players_xy[players_for_pitch.class_id == 1],
        face_color=sv.Color.from_hex(f"#{cfg.color_team_1}"),
        edge_color=sv.Color.BLACK, radius=16, pitch=radar,
    )
    radar = draw_points_on_pitch(
        config=pitch_config, xy=pitch_referees_xy,
        face_color=sv.Color.from_hex(f"#{cfg.color_referee}"),
        edge_color=sv.Color.BLACK, radius=16, pitch=radar,
    )
    sv.plot_image(radar)

    # ── Voronoi diagram ────────────────────────────────────────────────────
    voronoi = draw_pitch(pitch_config)
    voronoi = draw_pitch_voronoi_diagram(
        config=pitch_config,
        team_1_xy=pitch_players_xy[players_for_pitch.class_id == 0],
        team_2_xy=pitch_players_xy[players_for_pitch.class_id == 1],
        team_1_color=sv.Color.from_hex(f"#{cfg.color_team_0}"),
        team_2_color=sv.Color.from_hex(f"#{cfg.color_team_1}"),
        pitch=voronoi,
    )
    sv.plot_image(voronoi)

    # ── White-background blend ─────────────────────────────────────────────
    blend = draw_pitch(pitch_config, background_color=sv.Color.WHITE, line_color=sv.Color.BLACK)
    blend = draw_points_on_pitch(
        config=pitch_config, xy=pitch_ball_xy,
        face_color=sv.Color.WHITE, edge_color=sv.Color.WHITE, radius=8, thickness=1, pitch=blend,
    )
    blend = draw_points_on_pitch(
        config=pitch_config, xy=pitch_players_xy[players_for_pitch.class_id == 0],
        face_color=sv.Color.from_hex(f"#{cfg.color_team_0}"),
        edge_color=sv.Color.WHITE, radius=16, thickness=1, pitch=blend,
    )
    blend = draw_points_on_pitch(
        config=pitch_config, xy=pitch_players_xy[players_for_pitch.class_id == 1],
        face_color=sv.Color.from_hex(f"#{cfg.color_team_1}"),
        edge_color=sv.Color.WHITE, radius=16, thickness=1, pitch=blend,
    )
    sv.plot_image(blend)
