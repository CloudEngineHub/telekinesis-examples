"""Demonstrates the Telekinesis StereoImages datatype."""

import time

import numpy as np
from loguru import logger

from telekinesis import datatypes


def stereo_images_example():
    """Demonstrate creation, inspection, operations, and serialization."""

    # ======================= Create ============================================
    left_pixels = np.arange(480 * 640, dtype=np.uint8).reshape(480, 640)
    right_pixels = np.roll(left_pixels, shift=8, axis=1)
    color_pixels = np.repeat(left_pixels[..., np.newaxis], repeats=3, axis=2)
    stereo_images = datatypes.StereoImages(
        left_image=datatypes.Image(left_pixels),
        right_image=datatypes.Image(right_pixels),
        color_image=datatypes.Image(color_pixels),
        timestamp=datatypes.Timestamp(sec=1, nanosec=234_500_000),
        frame_number=42,
    )
    logger.info(f"Created StereoImages: {stereo_images}")

    # ======================= Inspect ===========================================
    logger.info(f"left image shape={stereo_images.left_image.shape}")
    logger.info(f"right image shape={stereo_images.right_image.shape}")
    logger.info(f"color image shape={stereo_images.color_image.shape}")
    logger.info(f"timestamp={stereo_images.timestamp}")
    logger.info(f"frame number={stereo_images.frame_number}")

    # ======================= Operations ========================================
    returned_left_image = stereo_images.left_image
    returned_left_image.data = np.zeros_like(left_pixels)
    logger.info(
        "Mutating the returned left image does not affect the observation: "
        f"first pixel={stereo_images.left_image.data[0, 0]}"
    )

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(stereo_images)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized StereoImages: {deserialized}")
    logger.info(f"Round-trip successful: {stereo_images == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    stereo_images_example()
