from pathlib import Path
from typing import List, Optional, Callable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from visual_pose.data_utils.pose_graph_samples import SAMPLE_FILE, is_current


_UPPER = torch.triu_indices(6, 6)   # the 21 entries of a symmetric 6x6 that are not duplicates


def edge_features(Z: np.ndarray, info: np.ndarray) -> Tensor:
    """(E, 33): Z's top three rows (12; the bottom row is always 0 0 0 1), then info's upper triangle (21)."""
    Z, info = torch.from_numpy(Z).float(), torch.from_numpy(info).float()
    return torch.cat([Z[:, :3].flatten(1), info[:, _UPPER[0], _UPPER[1]]], dim=1)


class PoseGraphDataset(Dataset):

    def __init__(
            self,
            paths_to_trajectories: List[Path],
            descriptor_gen: Optional[Callable],

    ):
        # samples are built offline (python -m visual_pose.data_utils.pose_graph_samples), never here
        stale = [p for p in paths_to_trajectories if not is_current(p)]
        if stale:
            raise FileNotFoundError(f"{len(stale)} trajectories lack a current {SAMPLE_FILE}, e.g. {stale[0]}: "
                                    "run visual_pose.data_utils.pose_graph_samples first")
        self.descriptor_gen = descriptor_gen
        self.sample_paths = [p / SAMPLE_FILE for p in paths_to_trajectories]

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, index) -> dict[str, Optional[Tensor]]:
        # with: an .npz stays open until garbage collection otherwise, leaking handles across epochs and workers
        with np.load(self.sample_paths[index]) as data:
            if self.descriptor_gen and 'descriptors' not in data:
                raise KeyError(f"{self.sample_paths[index]} has no descriptors: pose_graph_samples does not write them yet")
            # CPU tensors: DataLoader workers must not touch the GPU; the training step moves the batch
            return {
                'poses': torch.from_numpy(data['poses']).float(),  # (V, 4, 4) current best guess for trajectory: chained odometry, vertex 0 = identity
                'gt_poses': torch.from_numpy(data['gt_poses']).float(),  # (V, 4, 4) ground truth trajectory, anchored to vertex 0. TARGET ONLY
                'points': torch.from_numpy(data['points']).float(),  # (V, P, 3) camera frame, metres
                'closures': edge_features(data['closures_Z'], data['closure_info']),  # (L, 12 + 21)
                'closures_ij': torch.from_numpy(data['closures_ij']).long(),  # (L, 2)
                'odometry': edge_features(data['odometry_Z'], data['odometry_info']),  # (O, 12 + 21)
                'odometry_ij': torch.from_numpy(data['odometry_ij']).long(),  # (O, 2)
                'descriptors': torch.from_numpy(data['descriptors']).float() if self.descriptor_gen else None,
            }
