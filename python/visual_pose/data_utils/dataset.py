from abc import ABC, abstractmethod
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from visual_pose.data_utils.constants import INTRINSICS_TARTAN_AIR

R_ned_cv = np.array([
    [0, 0, 1],
    # NED_x = CV_z
    [1, 0, 0],
    # NED_y = CV_x
    [0, 1, 0],
    # NED_z = CV_y
], dtype=np.float32)
T_ned_cv = np.eye(4)
T_ned_cv[:3, :3] = R_ned_cv

class FrameSequence(ABC):
    """
    A sequence of posed RGB-D frames. Consumers (correspondence datasets,
    run.py) see only this. Each implementation converts its own camera
    convention so that pose(i) is T_WC with OpenCV axes and K matches.
    """
    K: np.ndarray            # (3, 3)
    # True when the frames are already spaced for the estimator (the HM3D
    # generator emits one frame per 0.6 m / 13.5 deg), so the pipeline runs
    # on every frame; a video-rate sequence like TartanAir is subsampled by
    # Config.val_frame_gap instead.
    frames_are_keyframes: bool = False
    # False when pose(i) is a placeholder (a raw recording): run.py then skips
    # every error and ATE, and draws no reference.
    has_ground_truth: bool = True

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def rgb(self, i: int) -> np.ndarray:
        """(H, W, 3) uint8, RGB."""

    @abstractmethod
    def depth(self, i: int) -> np.ndarray:
        """(H, W) float32 metres; invalid pixels are dataset-specific, callers filter with usable_depth."""

    @abstractmethod
    def pose(self, i: int) -> np.ndarray:
        """(4, 4) T_WC, OpenCV camera axes."""


class ArraySequence(FrameSequence):
    """
    Frames already in memory: what a renderer or a bag reader produces.
    rgb (N, H, W, 3) uint8; depth (N, H, W) float32 metres, 0 = no return;
    poses (N, 4, 4) T_WC in OpenCV axes -- the caller converts its camera
    convention (HM3DSequence, bag_sequence.read_bag) so this class has none.
    """

    def __init__(self, rgb: np.ndarray, depth: np.ndarray, poses: np.ndarray, K: np.ndarray, *,
                 frames_are_keyframes: bool = False, has_ground_truth: bool = True):
        self._rgb = np.asarray(rgb)
        self._depth = np.asarray(depth, dtype=np.float32)
        self.poses = np.asarray(poses, dtype=np.float64).reshape(-1, 4, 4)
        self.K = np.asarray(K, dtype=np.float32)
        self.frames_are_keyframes = frames_are_keyframes
        self.has_ground_truth = has_ground_truth
        assert len(self._rgb) == len(self._depth) == len(self.poses)
        assert self._rgb.shape[1:3] == self._depth.shape[1:3], (self._rgb.shape, self._depth.shape)

    def __len__(self) -> int:
        return len(self.poses)

    def rgb(self, i: int) -> np.ndarray:
        return self._rgb[i]

    def depth(self, i: int) -> np.ndarray:
        return self._depth[i]

    def pose(self, i: int) -> np.ndarray:
        return self.poses[i]


class TartanAirSequence(FrameSequence):
    def __init__(self, dataset_dir: Path):
        """
        Initialize a TartanAirSequence object.

        self.poses: (N, 4, 4) transforms poses from camera frame to world frame.

        :param dataset_dir:
        """
        self.dataset_dir = Path(dataset_dir)
        self.image_paths = sorted((self.dataset_dir / "image_left").glob("*_left.png"))
        self.depth_paths = sorted((self.dataset_dir / "depth_left").glob("*_left_depth.npy"))
        self.poses = self._load_poses()
        self.K = INTRINSICS_TARTAN_AIR

        assert len(self.image_paths) == len(self.depth_paths)
        assert len(self.image_paths) == len(self.poses)

    def _load_poses(self) -> np.ndarray:
        """(N, 4, 4) T_WC, camera frame into world, OpenCV axes."""
        poses_raw = np.loadtxt(self.dataset_dir / "pose_left.txt")
        translations = poses_raw[:, :3]
        quats = poses_raw[:, 3:]

        rotations = R.from_quat(quats).as_matrix()
        poses = np.tile(np.eye(4), (len(poses_raw), 1, 1))

        poses[:, :3, :3] = rotations
        poses[:, :3, 3] = translations
        poses_corrected = poses @ T_ned_cv
        return poses_corrected

    def __len__(self) -> int:
        return len(self.image_paths)

    def rgb(self, i: int) -> np.ndarray:
        """(H, W, 3) uint8, RGB."""
        bgr = cv2.imread(str(self.image_paths[i]))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def depth(self, i: int) -> np.ndarray:
        """(H, W) float32 metres. ~1e4 for sky -- callers must filter."""
        return np.load(str(self.depth_paths[i]))

    def pose(self, i: int) -> np.ndarray:
        """(4, 4) T_WC."""
        return self.poses[i]