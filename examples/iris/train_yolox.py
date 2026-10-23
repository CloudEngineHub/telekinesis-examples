"""Train a YOLOX detector on a COCO dataset."""

import argparse
from pathlib import Path

import torch
from loguru import logger

from telekinesis.iris.dataset import COCODataset, prepare_coco_datasets
from telekinesis.iris.logger import TensorBoardMetricLogger
from telekinesis.iris.models import YOLOX, YOLOXConfig, YOLOXVariant
from telekinesis.iris.optimizer import OptimizationConfig
from telekinesis.iris.trainer import Precision, Trainer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="yolox")
    parser.add_argument(
        "--variant",
        choices=[variant.value for variant in YOLOXVariant],
        default=YOLOXVariant.SMALL.value,
    )
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--nms-threshold", type=float, default=0.65)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--no-aug-epochs", type=int, default=15)
    parser.add_argument("--warmup-epochs", type=float, default=5.0)
    parser.add_argument("--validation-split", type=float, default=0.3)
    parser.add_argument("--evaluation-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="For example: cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--precision",
        choices=[precision.value for precision in Precision],
        default=Precision.FP32.value,
    )
    parser.add_argument("--resume-from", type=Path)
    return parser.parse_args()


def train_yolox(args: argparse.Namespace) -> None:
    """Configure and train YOLOX."""
    metadata_dir = args.dataset_dir / "train"
    if not metadata_dir.is_dir():
        metadata_dir = args.dataset_dir
    metadata = COCODataset(metadata_dir)
    torch.manual_seed(args.seed)

    config = YOLOXConfig(
        variant=YOLOXVariant(args.variant),
        num_classes=len(metadata.category_ids),
        category_ids=tuple(metadata.category_ids),
        pretrained=args.pretrained,
        resolution=args.resolution,
        nms_threshold=args.nms_threshold,
        max_detections=args.max_detections,
        no_aug_epochs=args.no_aug_epochs,
        warmup_epochs=args.warmup_epochs,
        optimization=OptimizationConfig(
            optimizer="sgd",
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            optimizer_options={"momentum": 0.9, "nesterov": True},
            scheduler="cosine",
            scheduler_options={"min_factor": 0.05},
            scheduler_interval="step",
        ),
    )
    model = YOLOX(config=config)
    training_dataset, validation_dataset = prepare_coco_datasets(
        dataset_dir=args.dataset_dir,
        validation_split=args.validation_split,
        seed=args.seed,
        train_transforms=model.train_transforms,
        val_transforms=model.val_transforms,
        include_masks=model.requires_masks,
    )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training YOLOX on {len(training_dataset)} images using {device}.")
    metric_logger = TensorBoardMetricLogger(args.output_dir / "tensorboard")
    model_trainer = Trainer(
        model=model,
        run_name=args.run_name,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
        precision=args.precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_interval=args.evaluation_interval,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
        metric_logger=metric_logger,
    )
    if args.resume_from is None:
        model_trainer.train()
    else:
        model_trainer.resume(args.resume_from)


if __name__ == "__main__":
    train_yolox(parse_args())
