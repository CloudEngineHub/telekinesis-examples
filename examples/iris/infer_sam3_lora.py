"""Run local SAM3 LoRA inference on one image."""

import argparse
from pathlib import Path

import rerun as rr
from loguru import logger

from telekinesis.iris.deploy import Model, SAM3PostprocessConfig
from telekinesis.iris.visualizer import visualize_prediction


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompts", nargs="+")
    parser.add_argument("--device", help="For example: cuda, cuda:0, or cpu.")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--per-prompt-nms-threshold", type=float, default=0.5)
    parser.add_argument(
        "--prompt-conflict-policy",
        choices=("multi_label", "best_prompt"),
        default="multi_label",
    )
    parser.add_argument("--cross-prompt-iou-threshold", type=float, default=0.8)
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def infer_sam3_lora(args: argparse.Namespace) -> None:
    """Load the exported bundle and predict one image locally."""
    model = Model(args.artifact, device=args.device)
    prediction = model.predict(
        args.image,
        confidence_threshold=args.confidence_threshold,
        prompts=args.prompts,
        max_detections=args.max_detections,
        postprocess=SAM3PostprocessConfig(
            per_prompt_nms_threshold=args.per_prompt_nms_threshold,
            prompt_conflict_policy=args.prompt_conflict_policy,
            cross_prompt_iou_threshold=args.cross_prompt_iou_threshold,
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
        rr.init("sam3_lora_inference", spawn=True)
        visualize_prediction(args.image, prediction)


if __name__ == "__main__":
    infer_sam3_lora(parse_args())
