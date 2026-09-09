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

class TartanAirSequence:
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