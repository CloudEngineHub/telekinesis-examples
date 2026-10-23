"""Run local RF-DETR inference on one image."""

import argparse
from pathlib import Path

import rerun as rr
from loguru import logger

from telekinesis.iris.deploy import Model, RFDETRPostprocessConfig
from telekinesis.iris.visualizer import visualize_prediction


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--class-names", nargs="+")
    parser.add_argument("--providers", nargs="+")
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--background-class-id", type=int)
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def infer_rfdetr(args: argparse.Namespace) -> None:
    """Load the exported artifact and predict one image locally."""
    model = Model(
        args.artifact,
        class_names=args.class_names,
        providers=args.providers,
    )
    prediction = model.predict(
        args.image,
        confidence_threshold=args.confidence_threshold,
        max_detections=args.max_detections,
        postprocess=RFDETRPostprocessConfig(
            background_class_id=args.background_class_id,
        ),
    )
    logger.info(f"Found {len(prediction)} detections in {args.image.name}")
    for box, score, class_id, class_name in zip(
        prediction.boxes,
        prediction.scores,
        prediction.class_ids,
        prediction.class_names or map(str, prediction.class_ids),
        strict=True,
    ):
        logger.info(
            f"class={class_name} id={class_id} score={score:.3f} "
            f"box_xywh={box.round(1).tolist()}"
        )
    if args.visualize:
        rr.init("rfdetr_inference", spawn=True)
        visualize_prediction(args.image, prediction)


if __name__ == "__main__":
    infer_rfdetr(parse_args())
