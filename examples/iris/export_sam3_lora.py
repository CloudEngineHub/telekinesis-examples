"""Export a SAM3 LoRA checkpoint to an Iris PyTorch bundle."""

import argparse
from pathlib import Path

from loguru import logger

from telekinesis.iris.export import export_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-name", default="model")
    return parser.parse_args()


def export_sam3_lora(args: argparse.Namespace) -> None:
    """Export the checkpoint."""
    artifact = export_model(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        format="bundle",
        artifact_name=args.artifact_name,
    )
    logger.success(f"Exported SAM3 LoRA bundle to {artifact.resolve()}")


if __name__ == "__main__":
    export_sam3_lora(parse_args())
