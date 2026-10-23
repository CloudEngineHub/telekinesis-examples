"""Train an RF-DETR model on a COCO dataset."""

import argparse
from pathlib import Path

import torch
from loguru import logger

from telekinesis.iris.dataset import AugmentationPreset, COCODataset, prepare_coco_datasets
from telekinesis.iris.logger import TensorBoardMetricLogger
from telekinesis.iris.models import RFDETR, RFDETRConfig, RFDETRVariant
from telekinesis.iris.trainer import Precision, Trainer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="rfdetr")
    parser.add_argument(
        "--variant",
        choices=[variant.value for variant in RFDETRVariant],
        default=RFDETRVariant.SEG_MEDIUM.value,
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--encoder-learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--learning-rate-drop-epochs", type=int, default=100)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--validation-split", type=float, default=0.3)
    parser.add_argument("--evaluation-interval", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="For example: cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--precision",
        choices=[precision.value for precision in Precision],
        default=Precision.AUTO.value,
    )
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from", type=Path)
    return parser.parse_args()


def train_rfdetr(args: argparse.Namespace) -> None:
    """Configure and train RF-DETR."""
    metadata_dir = args.dataset_dir / "train"
    if not metadata_dir.is_dir():
        metadata_dir = args.dataset_dir
    metadata = COCODataset(metadata_dir)
    torch.manual_seed(args.seed)

    config = RFDETRConfig(
        variant=RFDETRVariant(args.variant),
        num_classes=metadata.num_classes,
        pretrained=args.pretrained,
        resolution=args.resolution,
        learning_rate=args.learning_rate,
        encoder_learning_rate=args.encoder_learning_rate,
        weight_decay=args.weight_decay,
        learning_rate_drop_epochs=args.learning_rate_drop_epochs,
        augmentation=AugmentationPreset.RFDETR_EXTRA,
    )
    model = RFDETR(config=config)
    training_dataset, validation_dataset = prepare_coco_datasets(
        dataset_dir=args.dataset_dir,
        validation_split=args.validation_split,
        seed=args.seed,
        train_transforms=model.train_transforms,
        val_transforms=model.val_transforms,
        include_masks=model.requires_masks,
    )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        f"Training RF-DETR {config.variant.value} on {len(training_dataset)} images "
        f"using {device}."
    )
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
    train_rfdetr(parse_args())
