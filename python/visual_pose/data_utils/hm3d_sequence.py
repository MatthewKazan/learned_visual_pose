"""
HM3D frames as a FrameSequence, from arrays already rendered by habitat-sim.

    HM3DSequence(rgb, depth, poses_hab)     rgb (N,H,W,3) uint8, depth (N,H,W) float32, poses (N,4,4)

Camera axes. Habitat's camera is +X right, +Y up, looking down -Z; OpenCV's is
+X right, +Y down, looking down +Z. The fixed change of camera frame is a flip
of Y and Z, applied on the right of T_WC exactly as T_ned_cv is for TartanAir:
a point p_cv in the OpenCV camera frame is T_hab_cv p_cv in Habitat's, and the
world then sees T_WC_hab (T_hab_cv p_cv). Habitat depth is planar (the camera-Z
coordinate of the surface, not ray length), so it is OpenCV Z unchanged.

Verified 2026-09-28 on 00801: render a frame, move the camera 0.3 m along its
own axis, re-render, push the first frame's grid through both poses and compare
predicted depth with the second frame's: median |error| 1 mm and 98% within
5 cm with this conversion; 0.64 m median and 18% without it. Bottom-centre
pixel over a floor 1.5 m down reads 2.08 m, the planar prediction (Euclidean
would be 2.5 m).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from visual_pose.data_utils.constants import REPO_DIR
from visual_pose.data_utils.dataset import ArraySequence

HM3D = REPO_DIR / "data" / "hm3d" / "hm3d-minival-habitat-v0.2"
POSES_FILE = "poses.txt"

T_hab_cv = np.diag([1.0, -1.0, -1.0, 1.0])
HFOV_DEG = 90.0   # habitat CameraSensorSpec default
IMAGE_SIZE = (480, 640)   # (H, W) for every HM3D render: generator, GUI and run.py


def intrinsics_from_hfov(hfov_deg: float, H: int, W: int) -> np.ndarray:
    """Square pixels, principal point at the image centre: fx = (W/2) / tan(hfov/2)."""
    fx = (W / 2) / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], dtype=np.float32)


def simulator_config(scene_dir: Path, image_size: tuple[int, int] = IMAGE_SIZE):
    """
    habitat_sim.Configuration for one HM3D scene: RGB and depth cameras of
    `image_size` at the AGENT (zero sensor offset), so the agent state IS the
    camera pose and poses.txt, which carries the eye height, is rendered exactly.
    The one place the sensor rig is defined; generator, GUI and run.py share it.
    """
    import habitat_sim
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(next(Path(scene_dir).glob("*.basis.glb")))
    # minival ships no dataset config; habitat then builds a default one and
    # still finds <scene>.basis.navmesh by name
    dataset_cfg = next(Path(scene_dir).parent.glob("*.scene_dataset_config.json"), None)
    if dataset_cfg:
        sim_cfg.scene_dataset_config_file = str(dataset_cfg)
    specs = []
    for uuid, kind in (("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH)):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = kind
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        spec.resolution = list(image_size)
        spec.hfov = HFOV_DEG
        spec.position = [0.0, 0.0, 0.0]
        specs.append(spec)
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = specs
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def load_poses(path: Path) -> np.ndarray:
    """(N, 4, 4) T_WC in Habitat axes from a trajectory's poses.txt: one pose per line, 16 numbers row-major."""
    if not path.exists():
        return np.empty((0, 4, 4), dtype=np.float64)
    return np.loadtxt(path, dtype=np.float64, ndmin=2).reshape(-1, 4, 4)


@dataclass
class Trajectory:
    """One generated trajectory: <scene>/trajectories/<name>/poses.txt."""
    file: Path                # the trajectory directory
    scene_dir: Path           # where the scene's .glb and .navmesh are
    poses: np.ndarray         # (N, 4, 4) T_WC, Habitat axes, as load_poses reads them

    @property
    def points(self) -> np.ndarray:
        """Camera positions extracted from T_WC."""
        return self.poses[:, :3, 3]

    @property
    def label(self) -> str:
        return f"{self.scene_dir.name} / {self.file.name}"


def scene_assets(traj_dir: Path) -> Path:
    """
    The directory holding the scene's .glb and .navmesh. Normally the
    trajectory's grandparent; a trajectory generated with --out lives under
    OUT/<scene>/trajectories/, so fall back to the scene of that name in the
    HM3D data.
    """
    local = traj_dir.parents[1]
    if any(local.glob("*.navmesh")):
        return local
    return HM3D / local.name


def load_trajectory(traj_dir: Path) -> Trajectory:
    return Trajectory(file=traj_dir, scene_dir=scene_assets(traj_dir),
                      poses=load_poses(traj_dir / POSES_FILE))


def trajectory_dirs(root: Path = HM3D) -> list[Path]:
    """Every <scene>/trajectories/<name>/ under root that has a poses.txt."""
    return sorted(d for d in root.rglob("*/trajectories/*/") if (d / POSES_FILE).exists())


class HM3DSequence(ArraySequence):
    def __init__(self, rgb: np.ndarray, depth: np.ndarray, poses_hab: np.ndarray, hfov_deg: float = HFOV_DEG):
        """
        rgb (N, H, W, 3) uint8; depth (N, H, W) float32 metres, 0 = no return;
        poses_hab (N, 4, 4) T_WC in Habitat camera axes, as poses.txt stores them.
        """
        depth = np.asarray(depth, dtype=np.float32)
        super().__init__(rgb, depth,
                         np.asarray(poses_hab, dtype=np.float64).reshape(-1, 4, 4) @ T_hab_cv,
                         intrinsics_from_hfov(hfov_deg, *depth.shape[1:]),
                         frames_are_keyframes=True)   # generator spacing is keyframe spacing; see FrameSequence

