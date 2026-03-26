"""Mode 6 — Cluster player crops into two teams using pre-saved SigLIP embeddings."""

import numpy as np
import supervision as sv
from sklearn.cluster import KMeans
from tqdm import tqdm
import umap

from football_vision.config import AppConfig
from football_vision.core.video import get_frame_generator


def collect_crops(cfg: AppConfig, models: dict) -> list:
    """Iterate the source video with stride and collect player image crops."""
    frame_gen = get_frame_generator(cfg.source_video_path, stride=cfg.embedding_stride)
    crops = []
    for frame in tqdm(frame_gen, desc="collecting crops"):
        result = models["player"].infer(frame, confidence=cfg.detection_confidence)[0]
        detections = sv.Detections.from_inference(result)
        detections = detections.with_nms(threshold=cfg.nms_threshold, class_agnostic=True)
        detections = detections[detections.class_id == cfg.player_id]
        crops += [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
    return crops


def cluster_from_embeddings(
    embeddings_path: str, crops: list, n_components: int, n_teams: int
):
    """Load saved embeddings, reduce with UMAP, cluster with KMeans.

    Returns:
        team0_crops, team1_crops: lists of crop images per team
    """
    data = np.load(embeddings_path)

    reducer = umap.UMAP(n_components=n_components)
    projections = reducer.fit_transform(data)

    clustering = KMeans(n_clusters=n_teams)
    cluster_labels = clustering.fit_predict(projections)

    team0 = [crop for crop, label in zip(crops, cluster_labels) if label == 0]
    team1 = [crop for crop, label in zip(crops, cluster_labels) if label == 1]
    return team0, team1


def run(cfg: AppConfig, models: dict) -> None:
    crops = collect_crops(cfg, models)
    sv.plot_images_grid(crops[:100], grid_size=(10, 10))

    team0, team1 = cluster_from_embeddings(
        cfg.embeddings_path, crops, cfg.umap_components, cfg.n_teams
    )
    sv.plot_images_grid(team0[:100], grid_size=(10, 10))
    sv.plot_images_grid(team1[:100], grid_size=(10, 10))
