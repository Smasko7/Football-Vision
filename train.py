"""FootballVision — Training entry point.

Fine-tunes a YOLO model on a custom dataset and exports it to models/.

Usage examples:
    # Use settings from config.yaml
    python train.py

    # Override dataset and training parameters
    python train.py --dataset datasets/my_dataset/data.yaml --epochs 20 --device 0

    # Use a different base model
    python train.py --base-model models/yolo11s.pt --batch 8

After training, set trained_model_path in config.yaml to the printed export path
to use your model for inference with run.py.
"""

import argparse

from football_vision.config import load_config
from football_vision.core.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FootballVision — fine-tune YOLO on a custom dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Override dataset_path from config (path to data.yaml)",
    )
    parser.add_argument(
        "--base-model", default=None, dest="base_model",
        help="Override base_model from config (pretrained weights to fine-tune)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override training_epochs from config",
    )
    parser.add_argument(
        "--batch", type=int, default=None,
        help="Override training_batch from config",
    )
    parser.add_argument(
        "--device", default=None,
        help="Override training_device from config (e.g. 'cpu' or '0' for GPU)",
    )
    parser.add_argument(
        "--export-format", default=None, dest="export_format",
        help="Override export_format from config ('pt' or 'onnx')",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.dataset:
        cfg.dataset_path = args.dataset
    if args.base_model:
        cfg.base_model = args.base_model
    if args.epochs is not None:
        cfg.training_epochs = args.epochs
    if args.batch is not None:
        cfg.training_batch = args.batch
    if args.device:
        cfg.training_device = args.device
    if args.export_format:
        cfg.export_format = args.export_format

    train(cfg)


if __name__ == "__main__":
    main()
