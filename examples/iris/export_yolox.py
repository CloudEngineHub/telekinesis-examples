"""Export a YOLOX checkpoint to an ONNX artifact."""

import argparse
from pathlib import Path

from loguru import logger

from telekinesis.iris.export import YOLOXONNXExportOptions, export_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-name", default="model")
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--dynamic-batch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--opset-version", type=int, default=17)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def export_yolox(args: argparse.Namespace) -> None:
    """Export the checkpoint."""
    artifact = export_model(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        format="onnx",
        artifact_name=args.artifact_name,
        options=YOLOXONNXExportOptions(
            resolution=args.resolution,
            dynamic_batch=args.dynamic_batch,
            opset_version=args.opset_version,
            verbose=args.verbose,
        ),
    )
    logger.success(f"Exported YOLOX artifact to {artifact.resolve()}")


if __name__ == "__main__":
    export_yolox(parse_args())
