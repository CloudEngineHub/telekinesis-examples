"""Demonstrates the Telekinesis StereoCalibration datatype."""

import time

import numpy as np
from loguru import logger

from telekinesis import datatypes


def stereo_calibration_example():
    """Demonstrate creation, inspection, operations, and serialization."""

    # ======================= Create ============================================
    camera = datatypes.CameraCalibration(
        width=1280,
        height=720,
        distortion_model="plumb_bob",
        distortion_parameters=[-0.2, 0.1, 0.0, 0.0, 0.0],
        intrinsic_matrix=[1000.0, 0.0, 640.0, 0.0, 1000.0, 360.0, 0.0, 0.0, 1.0],
    )
    left_to_right = np.eye(4, dtype=np.float32)
    left_to_right[0, 3] = 0.12
    calibration = datatypes.StereoCalibration(
        left=camera,
        right=camera,
        left_to_right=datatypes.Mat4x4(left_to_right),
    )
    logger.info(f"Created StereoCalibration: {calibration}")

    # ======================= Inspect ===========================================
    logger.info(f"left calibration: {calibration.left}")
    logger.info(f"right calibration: {calibration.right}")
    logger.info(f"left-to-right transform:\n{calibration.left_to_right.to_numpy()}")
    logger.info(f"baseline={calibration.baseline:.3f} m")

    # ======================= Operations ========================================
    transform_copy = calibration.left_to_right
    transform_copy.data = np.eye(4, dtype=np.float32)
    logger.info(
        "Mutating the returned transform does not affect the calibration: "
        f"baseline={calibration.baseline:.3f} m"
    )

    # ======================= Serialize / Deserialize ===========================
    start = time.perf_counter()
    serialized = datatypes.serialize(calibration)
    serialization_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    deserialized = datatypes.deserialize(serialized)["param_0"]
    deserialization_ms = (time.perf_counter() - start) * 1000

    logger.info(f"Deserialized StereoCalibration: {deserialized}")
    logger.info(f"Round-trip successful: {calibration == deserialized}")
    logger.info(f"Serialization time: {serialization_ms:.3f} ms")
    logger.info(f"Deserialization time: {deserialization_ms:.3f} ms")


if __name__ == "__main__":
    stereo_calibration_example()
