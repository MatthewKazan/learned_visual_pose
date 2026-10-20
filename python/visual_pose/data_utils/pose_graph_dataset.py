from pathlib import Path
from typing import List, Optional, Callable

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import Tensor
from torch.utils.data import Dataset

from visual_pose import _geometry as cpp
from visual_pose.data_utils.pose_graph_samples import DESCRIPTOR_FILE, SAMPLE_FILE, is_current


_UPPER = torch.triu_indices(6, 6)   # the 21 entries of a symmetric 6x6 that are not duplicates


def pose_features(T: np.ndarray) -> Tensor:
    """(N, 12): each pose's top three rows, R | t row-major; the bottom row is always 0 0 0 1."""
    return torch.from_numpy(T).float()[:, :3].flatten(1)


def edge_features(Z: np.ndarray, info: np.ndarray) -> Tensor:
    """(E, 33): pose_features(Z) (12), then info's upper triangle (21)."""
    info = torch.from_numpy(info).float()
    return torch.cat([pose_features(Z), info[:, _UPPER[0], _UPPER[1]]], dim=1)


def delta_targets(T_in: np.ndarray, T_gt: np.ndarray) -> Tensor:
    """
    (V, 6) [rho, phi] of D_i = T_in_i^-1 T_gt_i, decoupled SO(3) x R^3: rho = D's translation,
    phi = Log(D's rotation) -- the encoding predicted_poses and correction() decode. Not the
    SE(3) Log, whose rho = V(phi)^-1 t. Per item, not stored: any augmentation that changes
    T_in changes the target.
    """
    def encode(D):
        return np.concatenate([D.translation(), np.asarray(D.rotation().log()).ravel()])
    return torch.from_numpy(np.stack([
        encode(cpp.PoseSE3(t).inverse() * cpp.PoseSE3(g))
        for t, g in zip(T_in, T_gt)])).float()


def delta_targets_mat(T_in: np.ndarray, T_gt: np.ndarray):
    return torch.from_numpy(np.stack([
        (cpp.PoseSE3(t).inverse() * cpp.PoseSE3(g)).matrix()
        for t, g in zip(T_in, T_gt)])).float()


def model_item(data, keep: slice, descriptors: Optional[Tensor] = None,
               closure_dropout: float = 0.0) -> dict[str, Optional[Tensor]]:
    """
    One graph's model inputs from the sample.npz arrays (an open .npz or sample_arrays' dict);
    the dataset and the live transformer solver both come through here. The TARGET ONLY entries
    exist only when data has gt_poses. CPU tensors: DataLoader workers must not touch the GPU.
    """
    item = {
        'poses': pose_features(data['poses']),  # (V, 12) current best guess for trajectory: chained odometry, vertex 0 = identity
        'points': torch.from_numpy(data['points'][:, keep]).float(),  # (V, P, 3) camera frame, metres
        'closures': edge_features(data['closures_Z'], data['closure_info']),  # (L, 12 + 21)
        'closures_ij': torch.from_numpy(data['closures_ij']).long(),  # (L, 2)
        'odometry': edge_features(data['odometry_Z'], data['odometry_info']),  # (O, 12 + 21)
        'odometry_ij': torch.from_numpy(data['odometry_ij']).long(),  # (O, 2)
        # fp16 as stored: half the transfer; the model casts. None when off
        'descriptors': descriptors,
        'closure_dropout': torch.rand(data['closures_ij'].shape[0]) < closure_dropout
    }
    if 'gt_poses' in data:
        item['gt_poses'] = torch.from_numpy(data['gt_poses']).float()  # (V, 4, 4) ground truth trajectory, anchored to vertex 0. TARGET ONLY
        item['delta_target'] = delta_targets(data['poses'], data['gt_poses'])  # (V, 6) [rho, phi] correction to poses. TARGET ONLY
        item['delta_target_mat'] = delta_targets_mat(data['poses'], data['gt_poses'])  # (V, 4, 4) correction to poses in matrix form. TARGET ONLY
    return item


class PoseGraphDataset(Dataset):

    def __init__(
            self,
            paths_to_trajectories: List[Path],
            pass_descriptor: bool = False,
            descriptor_dim: int = 384,
            num_points: Optional[int] = None,
            random_points: bool = True,
            closure_dropout: float = 0.0,
            pose_noise_deg: float = 0.0,
            pose_noise_m: float = 0.0,
    ):
        # samples are built offline (python -m visual_pose.data_utils.pose_graph_samples), never here
        stale = [p for p in paths_to_trajectories if not is_current(p)]
        if stale:
            raise FileNotFoundError(f"{len(stale)} trajectories lack a current {SAMPLE_FILE}, e.g. {stale[0]}: "
                                    "run visual_pose.data_utils.pose_graph_samples first")
        self.pass_descriptor = pass_descriptor
        self.sample_paths = [p / SAMPLE_FILE for p in paths_to_trajectories]
        self.descriptor_dim = descriptor_dim
        # Per item, a fresh random window of the stored points (and the same window of
        # their descriptors): per-point activations scale with B*V*P, ~4M points a
        # batch at 16 x 512 x 512. None keeps all.
        self.num_points = num_points
        # False: the first num_points every time, for a val score that does not
        # move between evaluations. Still a random sample -- sample_points stores
        # them in random order.
        self.random_points = random_points
        self.closure_dropout = closure_dropout
        self.pose_noise = (np.radians(pose_noise_deg), pose_noise_m)   # TransformerSolverConfig.pose_noise_*

    def __len__(self) -> int:
        return len(self.sample_paths)

    def _has_descriptors(self, index: int) -> bool:
        return (self.sample_paths[index].parent / DESCRIPTOR_FILE).exists()

    def _descriptors(self, index: int, keep: slice) -> np.ndarray:
        """
        (V, kept, D) float16. Stored (P, V, D), so the window is one contiguous block of the file.
        """
        header = np.load(self.sample_paths[index].parent / DESCRIPTOR_FILE, mmap_mode="r")
        row_items = int(np.prod(header.shape[1:]))
        start, stop, _ = keep.indices(header.shape[0])
        count = len(range(start, stop))
        offset = header.offset + start * row_items * header.itemsize

        window = np.fromfile(self.sample_paths[index].parent / DESCRIPTOR_FILE, dtype=header.dtype, count=count * row_items, offset=offset).reshape(count, *header.shape[1:])
        return np.ascontiguousarray(window.transpose(1, 0, 2))

    def __getitem__(self, index) -> dict[str, Optional[Tensor]]:
        # with: an .npz stays open until garbage collection otherwise, leaking handles across epochs and workers
        with np.load(self.sample_paths[index]) as data:
            P = data['points'].shape[1]
            # torch RNG, not numpy: DataLoader seeds it per worker, so workers draw different subsets
            if not self.num_points:
                keep = slice(None)
            elif self.random_points:
                # A window at a random offset, not a random subset: sample_points stores points in
                # random order, so a window is as random a draw, and in the memory-mapped
                # descriptors it is one contiguous read (P=512 -> 64: 1/8 of the bytes).
                start = int(torch.randint(max(P - self.num_points, 0) + 1, ()))
                keep = slice(start, start + self.num_points)
            else:
                keep = slice(0, self.num_points)
            descriptor_shape = (data['points'].shape[0], min(self.num_points or P, P), self.descriptor_dim)
            # None when off: no (V, P, D) zeros allocated and padded for nothing
            descriptors = ((torch.from_numpy(self._descriptors(index, keep)) if self._has_descriptors(index)
                            else torch.zeros(descriptor_shape, dtype=torch.float16)) if self.pass_descriptor else None)
            if any(self.pose_noise):
                data = dict(data)   # the input poses change, so do the targets model_item derives from them
                data['poses'] = perturb_poses(data['poses'], *self.pose_noise)
            return model_item(data, keep, descriptors, self.closure_dropout)


def perturb_poses(poses: np.ndarray, rot_rad: float, trans_m: float) -> np.ndarray:
    """(V,4,4) poses right-perturbed independently, T_i Exp: R_i Exp(N(0, rot_rad)), t_i + R_i N(0, trans_m);
    vertex 0, the gauge anchor, untouched. torch's RNG, which the DataLoader seeds per worker."""
    V = len(poses)
    rv, tv = (torch.randn(V, 3, dtype=torch.float64) * s for s in (rot_rad, trans_m))
    rv[0], tv[0] = 0.0, 0.0
    D = np.tile(np.eye(4), (V, 1, 1))
    D[:, :3, :3] = Rotation.from_rotvec(rv.numpy()).as_matrix()
    D[:, :3, 3] = tv.numpy()
    return poses @ D
