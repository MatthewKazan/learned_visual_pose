import numpy as np
import torch
from torch.utils.data import Dataset

from torchvision.transforms import ColorJitter

from visual_pose.data_utils.dataset import TartanAirSequence
from visual_pose.geometry.true_correspondences import generate_correspondences
from visual_pose.data_utils.constants import INTRINSICS_TARTAN_AIR, REPO_DIR


class TACorrespondenceDataset(Dataset):
    """
    Yields (rgb_i, rgb_j, uv_i, uv_j) per __getitem__, where uv_i and uv_j
    are matched pixel locations between the two frames.
    """

    def __init__(
            self,
            sequence: TartanAirSequence,
            frame_gap: int | list[int] = 5,
            num_correspondences: int = 512,
            min_correspondences: int = 128,
            darkness_threshold: float = 0.05,
            sample_step: int = 8,
            max_depth: float = 100.0,
            occlusion_tol: float = 0.5,
            K: np.ndarray = INTRINSICS_TARTAN_AIR,
            augment: bool = False,
            eval: bool = False,
            jitter: float = 0.2,
    ):
        self.seq = sequence
        self.num_correspondences = num_correspondences
        self.min_correspondences = min_correspondences
        self.sample_step = sample_step
        self.max_depth = max_depth
        self.occlusion_tol = occlusion_tol
        self.K = K
        self.eval = eval
        self.darkness_threshold = darkness_threshold

        # Photometric jitter only
        self.jitter = ColorJitter(brightness=jitter, contrast=jitter,
                                  saturation=jitter, hue=jitter / 8) if augment else None

        # candidate pairs (i, i + gap). Several gaps multiplies the pair count
        # from the same frames and spans a range of baselines, so the model sees
        # small and large apparent motion rather than one fixed amount.
        gaps = [frame_gap] if isinstance(frame_gap, int) else frame_gap
        self.pairs = [(i, i + g) for g in gaps for i in range(len(sequence) - g)]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index, _tries: int = 0):
        if len(self.pairs) <= index:
            raise IndexError(f"index {index} out of range for dataset of length {len(self)}")

        i, j = self.pairs[index]
        # randomize the grid phase per sample so the model sees every pixel offset,
        # not just multiples of sample_step. torch's RNG is seeded per DataLoader
        # worker; numpy's is not.
        grid_offset = tuple(torch.randint(0, self.sample_step, (2,)).tolist()) if not self.eval else (0,0)
        uv_i, uv_j = generate_correspondences(
            depth_i=self.seq.depth(i),
            depth_j=self.seq.depth(j),
            T_Wi=self.seq.pose(i),
            T_Wj=self.seq.pose(j),
            K=self.K,
            sample_step=self.sample_step,
            grid_offset=grid_offset,
            max_depth=self.max_depth,
            occlusion_tol=self.occlusion_tol,
        )

        rgb_i = self.seq.rgb(i)  # (H, W, 3) uint8
        rgb_j = self.seq.rgb(j)

        # to torch tensors, channel-first, normalized to [0, 1]
        img_i = torch.from_numpy(rgb_i).permute(2, 0, 1).float() / 255.0
        img_j = torch.from_numpy(rgb_j).permute(2, 0, 1).float() / 255.0
        uv_i = torch.from_numpy(uv_i).float()
        uv_j = torch.from_numpy(uv_j).float()

        clipped_uv_j = torch.round(uv_j)
        clipped_uv_j[:, 0] = torch.clamp(clipped_uv_j[:, 0], 0, img_j.shape[2] - 1)
        clipped_uv_j[:, 1] = torch.clamp(clipped_uv_j[:, 1], 0, img_j.shape[1] - 1)

        too_dark_mask = (img_i[:, uv_i[:, 1].long(), uv_i[:, 0].long()].mean(dim=0) < self.darkness_threshold) | (img_j[:, clipped_uv_j[:, 1].long(), clipped_uv_j[:, 0].long()].mean(dim=0) < self.darkness_threshold)

        if self.jitter is not None:
            img_i = self.jitter(img_i)
            img_j = self.jitter(img_j)   # separate draw, deliberately

        uv_i = uv_i[~too_dark_mask]
        uv_j = uv_j[~too_dark_mask]

        if len(uv_i) >= self.num_correspondences:
            if self.eval:
                rand = torch.randperm(len(uv_i), generator=torch.Generator().manual_seed(index))[:self.num_correspondences]
            else:
                rand = torch.randperm(len(uv_i))[:self.num_correspondences]
            uv_i = uv_i[rand]
            uv_j = uv_j[rand]
        elif _tries < 6:
            # Non-viable pairs cluster along a trajectory -- consecutive frames
            # are all sky, or all shadow -- so index+1 usually fails too and the
            # chain redoes two depth loads, two PNGs and the full geometry each
            # step. Measured 1296ms vs a 74ms median on P002. Jump a long way
            # instead, and cap the attempts.
            return self.__getitem__((index + 397) % len(self), _tries + 1)
        else:
            # Give up jumping and pad by resampling. Duplicates are bad for
            # InfoNCE (a duplicate row is a false negative at cosine 1.0) but
            # far better than an unbounded retry storm.
            pad = torch.randint(0, len(uv_i), (self.num_correspondences,))
            uv_i, uv_j = uv_i[pad], uv_j[pad]

        return {
            "rgb_i": img_i,
            "rgb_j": img_j,
            "uv_i":  uv_i,
            "uv_j":  uv_j,
        }