"""
One Source per data type, keyed as in run.py's RUNS.

To add one: subclass Source, set prefix / axes / max_depth, implement
discover, sequence and stamp (frames if the viewer should read something
other than sequence()), and put it in SOURCES before TartanAirSource.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from visual_pose.data_utils.bag_sequence import BAGS, read_bag
from visual_pose.data_utils.constants import REPO_DIR
from visual_pose.data_utils.dataset import FrameSequence, TartanAirSequence
from visual_pose.data_utils.hm3d_sequence import HM3D, POSES_FILE, T_hab_cv, load_poses, trajectory_dirs


class Source:
    prefix = ""
    axes = "RDF"             # rerun ViewCoordinates name of the sequence's world
    max_depth = 8.0          # metres; the viewer's map cutoff

    def __init__(self, key: str):
        self.key = key
        self.name = key[len(self.prefix):]

    @classmethod
    def discover(cls) -> list[str]:
        """Every key on disk."""
        return []

    def sequence(self) -> FrameSequence:
        """What run.py runs on."""
        raise NotImplementedError

    def frames(self) -> FrameSequence:
        """What the viewer draws."""
        return self.sequence()

    def stamp(self) -> float:
        """Changes when the data does."""
        return 0.0


class HM3DSource(Source):
    prefix, axes = "hm3d:", "RIGHT_HAND_Y_UP"

    def __init__(self, key: str):
        super().__init__(key)
        scene, trajectory = self.name.split("/")
        self.dir = HM3D / scene / "trajectories" / trajectory

    @classmethod
    def discover(cls) -> list[str]:
        return [f"{cls.prefix}{d.parents[1].name}/{d.name}" for d in trajectory_dirs()]

    def sequence(self) -> FrameSequence:
        from visual_pose.data_utils.hm3d_renderer import SceneRenderer
        renderer = SceneRenderer(self.dir.parents[1])
        try:
            return renderer.render_poses(load_poses(self.dir / POSES_FILE))
        finally:
            renderer.close()     # one GL context at a time; the model needs the GPU next

    def frames(self) -> FrameSequence:
        return SavedHM3DFrames(self.dir)

    def stamp(self) -> float:
        return (self.dir / POSES_FILE).stat().st_mtime


class SavedHM3DFrames(FrameSequence):
    """The generator's rgb/ and depth/, read per frame: a whole trajectory in memory is over 1 GB."""
    frames_are_keyframes = True

    def __init__(self, traj_dir: Path):
        from visual_pose.data_utils.hm3d_sequence import HFOV_DEG, IMAGE_SIZE, intrinsics_from_hfov
        self.dir = traj_dir
        self.poses = load_poses(traj_dir / POSES_FILE) @ T_hab_cv
        self.K = intrinsics_from_hfov(HFOV_DEG, *IMAGE_SIZE)

    def __len__(self) -> int:
        return len(self.poses)

    def rgb(self, i: int) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(str(self.dir / "rgb" / f"{i:06d}.jpg")), cv2.COLOR_BGR2RGB)

    def depth(self, i: int) -> np.ndarray:
        return cv2.imread(str(self.dir / "depth" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000

    def pose(self, i: int) -> np.ndarray:
        return self.poses[i]


class BagSource(Source):
    prefix, max_depth = "bag:", 5.0

    @classmethod
    def discover(cls) -> list[str]:
        return [f"{cls.prefix}{d.name}" for d in sorted(BAGS.glob("*/")) if any(d.glob("*.db3"))] \
            if BAGS.exists() else []

    def sequence(self) -> FrameSequence:
        return read_bag(BAGS / self.name)

    def stamp(self) -> float:
        return next((BAGS / self.name).glob("*.db3")).stat().st_mtime


class TartanAirSource(Source):
    axes, max_depth = "FRD", 30.0      # NED world; outdoors
    root = REPO_DIR / "data" / "tartan_air"

    @classmethod
    def discover(cls) -> list[str]:
        return [d.name for d in sorted(cls.root.glob("*/")) if (d / "pose_left.txt").exists()]

    def sequence(self) -> FrameSequence:
        return TartanAirSequence(self.root / self.name)

    def stamp(self) -> float:
        return (self.root / self.name / "pose_left.txt").stat().st_mtime


SOURCES = [HM3DSource, BagSource, TartanAirSource]     # TartanAir last: its prefix "" matches anything


def source(key: str) -> Source:
    return next(cls(key) for cls in SOURCES if key.startswith(cls.prefix))
