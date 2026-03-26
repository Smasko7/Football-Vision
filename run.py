"""FootballVision — CLI entry point.

Usage examples:
    python run.py --mode capture_frame
    python run.py --mode radar_video --source match.mp4 --target outputs/radar.mp4
    python run.py --mode basic_annotation --config custom_config.yaml

Available modes:
    capture_frame        Display the first frame of the source video
    basic_annotation     Bounding-box + confidence labels on a single frame
    videogame_frame      Video-game style ellipse/triangle annotation on a single frame
    videogame_video      Video-game style annotation written to a video file
    player_tracking      ByteTrack player tracking written to a video file
    team_split           Cluster player crops into two teams (requires saved embeddings)
    team_video           Per-frame team classification written to a video file
    pitch_keypoints      Detect and visualise pitch keypoints on a single frame
    radar_frame          Full pipeline (team classification + pitch radar) on a single frame
    radar_video          Full pipeline written to a video file
"""

import argparse
from pathlib import Path

from football_vision.config import load_config
from football_vision.core.models import load_field_model, load_player_model, load_yolo_model

import football_vision.modes.capture_frame as _capture_frame
import football_vision.modes.basic_annotation as _basic_annotation
import football_vision.modes.videogame_annotation as _videogame_annotation
import football_vision.modes.player_tracking as _player_tracking
import football_vision.modes.team_split as _team_split
import football_vision.modes.team_video as _team_video
import football_vision.modes.pitch_keypoints as _pitch_keypoints
import football_vision.modes.radar_frame as _radar_frame
import football_vision.modes.radar_video as _radar_video


# Which models each mode requires — only those are loaded at startup
_MODE_MODELS = {
    "capture_frame":     [],
    "basic_annotation":  ["player"],
    "videogame_frame":   ["player"],
    "videogame_video":   ["player"],
    "player_tracking":   ["player"],
    "team_split":        ["player"],
    "team_video":        ["player"],
    "pitch_keypoints":   ["field"],
    "radar_frame":       ["player", "field"],
    "radar_video":       ["player", "field"],
}

MODE_MAP = {
    "capture_frame":    _capture_frame.run,
    "basic_annotation": _basic_annotation.run,
    "videogame_frame":  lambda cfg, m: _videogame_annotation.run(cfg, m, as_video=False),
    "videogame_video":  lambda cfg, m: _videogame_annotation.run(cfg, m, as_video=True),
    "player_tracking":  _player_tracking.run,
    "team_split":       _team_split.run,
    "team_video":       _team_video.run,
    "pitch_keypoints":  _pitch_keypoints.run,
    "radar_frame":      _radar_frame.run,
    "radar_video":      _radar_video.run,
}

# Legacy display names → CLI key
_ALIASES = {
    "Capture a frame":                     "capture_frame",
    "Basic frame annotation":              "basic_annotation",
    "Video Game style frame annotation":   "videogame_frame",
    "Video annotation":                    "videogame_video",
    "Player tracking":                     "player_tracking",
    "Split into teams":                    "team_split",
    "Video for team classification":       "team_video",
    "Pitch keypoint detection":            "pitch_keypoints",
    "2D representations":                  "radar_frame",
    "2D representation radar view Video":  "radar_video",
}


def _resolve_mode(name: str) -> str:
    if name in MODE_MAP:
        return name
    if name in _ALIASES:
        return _ALIASES[name]
    raise ValueError(
        f"Unknown mode '{name}'.\nAvailable: {', '.join(MODE_MAP)}"
    )


def _build_models(mode_key: str, cfg) -> dict:
    needed = _MODE_MODELS[mode_key]
    models = {}
    if "player" in needed:
        if cfg.trained_model_path:
            print(f"Loading local YOLO model: {cfg.trained_model_path}")
            models["player"] = load_yolo_model(cfg.trained_model_path)
        else:
            print("Loading Roboflow player detection model…")
            models["player"] = load_player_model(
                cfg.player_detection_model_id, cfg.roboflow_api_key
            )
    if "field" in needed:
        print("Loading field detection model…")
        models["field"] = load_field_model(
            cfg.field_detection_model_id, cfg.roboflow_api_key
        )
    return models


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FootballVision — football video analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", required=True,
        help="Analysis mode to run (see available modes above)",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--source", default=None,
        help="Override source_video_path from config",
    )
    parser.add_argument(
        "--target", default=None,
        help="Override target_video_path from config",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.source:
        cfg.source_video_path = args.source
    if args.target:
        cfg.target_video_path = args.target

    mode_key = _resolve_mode(args.mode)

    # Auto-generate target path as outputs/<mode>/<source_stem>_<mode>.mp4
    # unless the user has already set one (via --target or config.yaml)
    if not cfg.target_video_path:
        source_stem = Path(cfg.source_video_path).stem
        Path(f"outputs/{mode_key}").mkdir(parents=True, exist_ok=True)
        cfg.target_video_path = f"outputs/{mode_key}/{source_stem}_{mode_key}.mp4"

    print(f"Source : {cfg.source_video_path}")
    print(f"Target : {cfg.target_video_path}")

    models = _build_models(mode_key, cfg)

    print(f"Running mode: {mode_key}")
    MODE_MAP[mode_key](cfg, models)


if __name__ == "__main__":
    main()
