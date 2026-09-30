"""Demonstrates generating 6-DOF grasps for a segmented object with GraspGenX.

GraspGenX serves any gripper from one model. The gripper is described purely
by its geometry: its link meshes at the fully-open and at the half-open state,
in the gripper's own base frame (+Z = approach axis, fingers closing along X
or Y). No gripper name, URDF or server-side asset is involved.

The helpers fetch a synapse tool's URDF bundle and place its link meshes
using forward kinematics to supply that geometry.

Returned poses are gripper-base poses in the point cloud's frame. The
`gripper_measurements` output tells you where the fingertips are: the fourth
value is the base-to-fingertip distance along each pose's +Z axis.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from loguru import logger
import rerun as rr
import trimesh
from scipy.spatial.transform import Rotation

from telekinesis import datatypes, vitreous
from telekinesis.synapse.tools.parallel_grippers import robotiq

ASSET_BASE_URL = "https://assets.telekinesis.ai"
NUM_GRASPS = 200


def load_object_point_cloud() -> datatypes.PointCloud:
    """Load hosted RGB-D assets and back-project the masked object in meters."""
    example_url = f"{ASSET_BASE_URL}/examples/v1"
    rgb = datatypes.Image.from_url(
        f"{example_url}/images/male_pin_connector_binpicking.png"
    ).data
    depth = datatypes.DepthImage.from_url(
        f"{example_url}/depth_images/male_pin_connector_binpicking.png",
        depth_scale=0.001,
    ).depth
    mask = datatypes.SegmentationImage.from_url(
        f"{example_url}/images/male_pin_connector_binpicking_mask.png"
    ).data
    # Same camera intrinsics as the FoundationPose example's RGB-D frame.
    intrinsic_matrix = np.array(
        [
            [435.46856689453125, 0.0, 420.7252502441406],
            [0.0, 434.5237731933594, 244.55136108398438],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    valid = (mask > 0) & np.isfinite(depth) & (depth > 0)
    rows, cols = np.nonzero(valid)
    if not len(rows):
        raise ValueError("The object mask contains no valid depth pixels.")
    pixels = np.column_stack((cols, rows, np.ones(len(rows))))
    positions = (pixels @ np.linalg.inv(intrinsic_matrix).T) * depth[valid, None]
    return datatypes.PointCloud(
        positions=positions.astype(np.float32),
        colors=rgb[valid],
    )


def fetch_tool_urdf(tool) -> datatypes.URDF:
    """Fetch the URDF bundle of a synapse tool from the asset server.

    Mirrors the naming convention synapse's `AbstractTool._build_model` uses:
    `urdf/tools/<tool type>/<brand>/<lowercased class name>.zip`.
    """
    module_parts = type(tool).__module__.split(
        "."
    )  # ...tools.parallel_grippers.robotiq
    tool_type, brand = module_parts[-2], module_parts[-1]
    bundle = type(tool).__name__.lower()
    return datatypes.URDF.from_url(
        f"{ASSET_BASE_URL}/urdf/tools/{tool_type}/{brand}/{bundle}.zip",
        use_cache=True,
        show_progress=False,
    )


def _origin_matrix(element) -> np.ndarray:
    """4x4 matrix of a URDF `<origin xyz rpy>` element (identity if absent)."""
    T = np.eye(4)
    if element is None:
        return T
    xyz = [float(v) for v in element.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in element.get("rpy", "0 0 0").split()]
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def _joint_motion(joint, value: float) -> np.ndarray:
    """4x4 motion of a URDF joint at `value` (radians or metres)."""
    axis = np.array(
        [
            float(v)
            for v in (
                joint.find("axis").get("xyz")
                if joint.find("axis") is not None
                else "1 0 0"
            ).split()
        ]
    )
    axis = axis / np.linalg.norm(axis)
    T = np.eye(4)
    kind = joint.get("type")
    if kind in ("revolute", "continuous"):
        T[:3, :3] = Rotation.from_rotvec(axis * value).as_matrix()
    elif kind == "prismatic":
        T[:3, 3] = axis * value
    return T


def gripper_link_meshes(
    urdf: datatypes.URDF, opening: float
) -> datatypes.Mesh3DBatch:
    """Link meshes of a parallel gripper at a given opening, in its base frame.

    Args:
        urdf: The gripper's URDF bundle.
        opening: Fraction of the actuated joint's travel from its lower to
            its upper limit, in `[0, 1]`. Which end is "open" depends on the
            gripper; see `pick_open_and_half_open`.

    Returns:
        A `Mesh3DBatch` with one mesh per link that has a visual mesh, in a
        fixed link order (the same for every `opening`).
    """
    urdf_path = Path(urdf.path)
    root = ET.parse(urdf_path).getroot()
    joints = root.findall("joint")
    child_to_joint = {j.find("child").get("link"): j for j in joints}

    # One actuated joint drives a parallel gripper; the others mimic it.
    actuated = [
        j
        for j in joints
        if j.get("type") in ("revolute", "prismatic", "continuous")
        and j.find("mimic") is None
    ]
    if len(actuated) != 1:
        raise ValueError(
            f"expected exactly one actuated joint in {urdf_path.name}, found {len(actuated)}"
        )
    drive = actuated[0]
    limit = drive.find("limit")
    lower, upper = float(limit.get("lower")), float(limit.get("upper"))
    drive_value = lower + opening * (upper - lower)

    def joint_value(joint) -> float:
        mimic = joint.find("mimic")
        if mimic is not None:
            return float(mimic.get("multiplier", 1.0)) * drive_value + float(
                mimic.get("offset", 0.0)
            )
        return drive_value if joint is drive else 0.0

    transforms: dict[str, np.ndarray] = {}

    def base_T_link(link_name: str) -> np.ndarray:
        if link_name in transforms:
            return transforms[link_name]
        joint = child_to_joint.get(link_name)
        if joint is None:  # root link
            transforms[link_name] = np.eye(4)
        else:
            parent = base_T_link(joint.find("parent").get("link"))
            motion = (
                np.eye(4)
                if joint.get("type") == "fixed"
                else _joint_motion(joint, joint_value(joint))
            )
            transforms[link_name] = (
                parent @ _origin_matrix(joint.find("origin")) @ motion
            )
        return transforms[link_name]

    meshes = []
    for link in root.findall("link"):
        visual = link.find("visual")
        mesh_element = (
            visual.find("geometry/mesh") if visual is not None else None
        )
        if mesh_element is None:
            continue
        mesh_file = mesh_element.get("filename").replace("package://", "")
        mesh = trimesh.load(urdf_path.parent / mesh_file, force="mesh")
        if mesh_element.get("scale"):
            mesh.apply_scale(
                [float(s) for s in mesh_element.get("scale").split()]
            )
        T = base_T_link(link.get("name")) @ _origin_matrix(
            visual.find("origin")
        )
        vertices = trimesh.transform_points(mesh.vertices, T).astype(np.float32)
        meshes.append(
            datatypes.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=np.asarray(mesh.faces, np.int32),
            )
        )
    return datatypes.Mesh3DBatch(meshes)


def pick_open_and_half_open(
    urdf: datatypes.URDF,
) -> tuple[datatypes.Mesh3DBatch, datatypes.Mesh3DBatch]:
    """Geometry at the fully-open and half-open states, whichever joint end is 'open'."""
    at_lower, at_upper, at_mid = (
        gripper_link_meshes(urdf, f) for f in (0.0, 1.0, 0.5)
    )

    def finger_spread(
        batch: datatypes.Mesh3DBatch, other: datatypes.Mesh3DBatch
    ) -> float:
        # extent across X/Y of the links that move between the two states
        moving = [
            a
            for a, b in zip(batch.vertex_positions, other.vertex_positions)
            if not np.allclose(a, b, atol=1e-6)
        ]
        pts = np.vstack(moving)
        return float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))

    is_lower_open = finger_spread(at_lower, at_upper) > finger_spread(
        at_upper, at_lower
    )
    return (at_lower if is_lower_open else at_upper), at_mid


def generate_grasps_using_graspgenx_example():
    """Generate and visualize grasps using a Robotiq gripper's geometry."""
    # ===================== Load Data ==========================================
    # Use the same hosted connector frame as the FoundationPose example.
    point_cloud = load_object_point_cloud()

    # The gripper: a synapse tool. Its geometry, not its name, is what gets sent.
    gripper = robotiq.Robotiq2F85()
    gripper_urdf = fetch_tool_urdf(gripper)
    gripper_open, gripper_half_open = pick_open_and_half_open(gripper_urdf)

    # ===================== Run Skill ==========================================
    grasps, scores, from_heuristic, gripper_measurements = (
        vitreous.generate_grasps_using_graspgenx(
            point_cloud,
            gripper_open,
            gripper_half_open,
            planner="graspmoe",
            num_grasps=NUM_GRASPS,
            topk_num_grasps=NUM_GRASPS,
        )
    )

    if len(grasps.data) == 0:
        raise RuntimeError("GraspGenX returned no grasps for this object.")

    # ===================== Log ================================================
    logger.success(f"Generated {len(scores.data)} grasps for {point_cloud}")
    logger.success(f"Results: {grasps}")
    measured = gripper_measurements.data
    logger.info(
        f"Gripper measured from geometry (m): opening {measured[0]:.3f}, "
        f"pad {measured[1]:.3f} x {measured[2]:.3f}, "
        f"fingertip depth {measured[3]:.3f}, half open {measured[4]:.3f}"
    )
    logger.info(f"Best grasp score: {scores.data[0]:.3f}")
    best = grasps.data[0]
    approach = best[:3, 2]
    fingertips = best[:3, 3] + measured[3] * approach
    logger.info(
        f"Best grasp (gripper base pose in the point cloud frame):\n{np.round(best, 4)}"
    )
    logger.info(
        f"Fingertip position: {np.round(fingertips, 4)} | "
        f"Approach direction: {np.round(approach, 3)}"
    )
    logger.info(
        f"Grasps from the bounding-box heuristic: {int(from_heuristic.data.sum())}"
    )

    # ===================== Visualization  (Optional) ===========================
    rr.init("generate_grasps_using_graspgenx_example", spawn=True)
    datatypes.visualize(point_cloud, entity_path="/1-point_cloud")
    datatypes.visualize(
        grasps,
        entity_path="/2-grasps",
        label=[f"{s:.2f}" for s in scores.data],
    )

    # Show every returned grasp with its own gripper geometry. Separate entity
    # paths let individual candidates be toggled in Rerun's entity tree.
    for index, grasp in enumerate(grasps.data):
        gripper_at_grasp = datatypes.Mesh3DBatch(
            [
                datatypes.Mesh3D(
                    vertex_positions=trimesh.transform_points(vertices, grasp).astype(
                        np.float32
                    ),
                    triangle_indices=triangles,
                )
                for vertices, triangles in zip(
                    gripper_open.vertex_positions, gripper_open.triangle_indices
                )
            ]
        )
        datatypes.visualize(
            gripper_at_grasp, entity_path=f"/3-grasp_grippers/grasp_{index:03d}"
        )


if __name__ == "__main__":
    generate_grasps_using_graspgenx_example()
