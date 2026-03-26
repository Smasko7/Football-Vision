"""YOLO fine-tuning logic for FootballVision."""

from pathlib import Path

from ultralytics import YOLO

from football_vision.config import AppConfig


def train(cfg: AppConfig) -> str:
    """Fine-tune a YOLO model on the dataset specified in cfg.

    Returns the path to the exported model file.
    """
    print(f"Base model  : {cfg.base_model}")
    print(f"Dataset     : {cfg.dataset_path}")
    print(f"Epochs      : {cfg.training_epochs}")
    print(f"Batch       : {cfg.training_batch}")
    print(f"Image size  : {cfg.training_imgsz}")
    print(f"Device      : {cfg.training_device}")
    print(f"Export fmt  : {cfg.export_format}")
    print()

    model = YOLO(cfg.base_model)

    model.train(
        data=cfg.dataset_path,
        epochs=cfg.training_epochs,
        batch=cfg.training_batch,
        imgsz=cfg.training_imgsz,
        device=cfg.training_device,
    )

    model.val()

    # Derive export name: <dataset_folder>_<base_model_stem>
    dataset_name = Path(cfg.dataset_path).parent.parent.name  # e.g. football-players-detection-dataset
    base_stem = Path(cfg.base_model).stem                     # e.g. yolo11n
    export_name = f"{dataset_name}_{base_stem}"
    export_path = str(Path("models") / f"{export_name}.{cfg.export_format}")

    model.export(format=cfg.export_format)

    # Move the exported file into models/ if YOLO placed it elsewhere
    default_export = Path(model.trainer.best).with_suffix(f".{cfg.export_format}")
    if default_export.exists() and not Path(export_path).exists():
        Path("models").mkdir(exist_ok=True)
        default_export.rename(export_path)

    print()
    print(f"Model exported to: {export_path}")
    print(f"To use this model for inference, set in config.yaml:")
    print(f'  trained_model_path: "{export_path}"')

    return export_path
