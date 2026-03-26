import os
from dataclasses import dataclass, field
from typing import List

import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AppConfig:
    # Video I/O
    source_video_path: str = "Demo_Video.mp4"
    target_video_path: str = "outputs/output.mp4"

    # Roboflow
    roboflow_api_key: str = ""
    player_detection_model_id: str = "football-players-detection-3zvbc/15"
    field_detection_model_id: str = "football-field-detection-f07vi/14"

    # Training
    dataset_path: str = "datasets/football-players-detection-dataset/versions/1/data.yaml"
    base_model: str = "models/yolo11n.pt"
    trained_model_path: str = ""
    training_epochs: int = 2
    training_batch: int = 2
    training_imgsz: int = 640
    training_device: str = "cpu"
    export_format: str = "pt"

    # Detection thresholds
    detection_confidence: float = 0.3
    nms_threshold: float = 0.5
    ball_pad_px: int = 10

    # Class IDs
    ball_id: int = 0
    goalkeeper_id: int = 1
    player_id: int = 2
    referee_id: int = 3

    # Team classifier
    siglip_model_path: str = "google/siglip-base-patch16-224"
    team_classifier_batch_size: int = 32
    umap_components: int = 3
    n_teams: int = 2
    embedding_stride: int = 30

    # Embeddings cache
    embeddings_path: str = "embeddings/siglip_embeddings.npy"

    # Per-mode frame limits
    annotation_frame_limit: int = 750
    tracking_frame_limit: int = 250
    radar_frame_limit: int = 250
    team_video_frame_limit: int = 10

    # Annotator colours
    color_team_0: str = "00BFFF"
    color_team_1: str = "FF1493"
    color_ball: str = "FFD700"
    color_referee: str = "FFD700"
    color_box_palette: List[str] = field(
        default_factory=lambda: ["FF8C00", "00BFFF", "FF1493", "FFD700"]
    )

    # Annotator geometry
    ellipse_thickness: int = 2
    triangle_base: int = 25
    triangle_height: int = 21
    triangle_outline_thickness: int = 1
    keypoint_radius: int = 8
    keypoint_confidence_threshold: float = 0.5


def load_config(path: str = "config.yaml") -> AppConfig:
    """Load AppConfig from a YAML file.

    The environment variable ROBOFLOW_API_KEY, if set, takes precedence
    over the value in the YAML file.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    cfg = AppConfig(**{k: v for k, v in data.items() if hasattr(AppConfig, k)})

    env_key = os.environ.get("ROBOFLOW_API_KEY")
    if env_key:
        cfg.roboflow_api_key = env_key

    return cfg
