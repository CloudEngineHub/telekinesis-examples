"""Fine-tune SAM3 with LoRA on a COCO dataset."""

import argparse
from pathlib import Path

import torch
from loguru import logger

from telekinesis.iris.dataset import COCODataset, prepare_coco_datasets
from telekinesis.iris.logger import TensorBoardMetricLogger
from telekinesis.iris.trainer import Precision, Trainer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="sam3_lora")
    parser.add_argument("--sam3-checkpoint", type=Path)
    parser.add_argument("--load-from-hf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--num-negative-prompts", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--gradient-clip-norm", type=float, default=0.1)
    parser.add_argument("--validation-split", type=float, default=0.3)
    parser.add_argument("--evaluation-interval", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="For example: cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--precision",
        choices=[precision.value for precision in Precision],
        default=Precision.AUTO.value,
    )
    parser.add_argument("--resume-from", type=Path)
    return parser.parse_args()


def train_sam3_lora(args: argparse.Namespace) -> None:
    """Configure and train SAM3 LoRA."""
    from telekinesis.iris.models import SAM3LoRA, SAM3LoRAConfig

    metadata_dir = args.dataset_dir / "train"
    if not metadata_dir.is_dir():
        metadata_dir = args.dataset_dir
    categories = COCODataset(metadata_dir).categories
    torch.manual_seed(args.seed)

    config = SAM3LoRAConfig(
        class_names=categories,
        sam3_checkpoint_path=args.sam3_checkpoint,
        load_from_hf=args.load_from_hf,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        apply_to_vision_encoder=True,
        apply_to_text_encoder=False,
        apply_to_geometry_encoder=True,
        apply_to_detr_encoder=True,
        apply_to_detr_decoder=True,
        apply_to_mask_decoder=True,
        resolution=args.resolution,
        num_negative_prompts=args.num_negative_prompts,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler="cosine",
        scheduler_interval="step",
        warmup_steps=args.warmup_steps,
        augmentation=[
            {"name": "random_scale_crop", "scale_range": (0.8, 1.2)},
            {"name": "horizontal_flip", "probability": 0.5},
            {
                "name": "photometric_distort",
                "probability": 0.8,
                "brightness": 0.35,
                "contrast": 0.35,
                "saturation": 0.3,
                "hue": 0.03,
                "grayscale_probability": 0.05,
            },
            {"name": "gamma", "probability": 0.4, "gamma_range": (0.7, 1.4)},
            {
                "name": "gaussian_blur",
                "probability": 0.2,
                "sigma_range": (0.2, 1.2),
            },
            {"name": "sharpness", "probability": 0.2, "factor_range": (0.5, 1.5)},
            {"name": "gaussian_noise", "probability": 0.3, "std_range": (0.005, 0.03)},
            {
                "name": "jpeg_compression",
                "probability": 0.3,
                "quality_range": (60, 100),
            },
        ],
    )
    model = SAM3LoRA(config=config)
    training_dataset, validation_dataset = prepare_coco_datasets(
        dataset_dir=args.dataset_dir,
        validation_split=args.validation_split,
        seed=args.seed,
        train_transforms=model.train_transforms,
        val_transforms=model.val_transforms,
        include_masks=model.requires_masks,
    )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training SAM3 LoRA on {len(training_dataset)} images using {device}.")
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
        gradient_clip_norm=args.gradient_clip_norm,
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
    train_sam3_lora(parse_args())
