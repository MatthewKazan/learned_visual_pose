from pathlib import Path
from typing import List, Optional, Callable

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import Tensor
from torch.utils.data import Dataset

from visual_pose import _geometry as cpp
from visual_pose.config.config import Config, TransformerSolverConfig
from visual_pose.data_utils.pose_graph_samples import DESCRIPTOR_FILE, SAMPLE_FILE, is_current


_UPPER = torch.triu_indices(6, 6)   # the 21 entries of a symmetric 6x6 that are not duplicates
# WorldModel.unknown_information: what the frontend gives an edge it could not verify
UNKNOWN_INFO = np.diag(1.0 / np.array([Config.unknown_edge_sigma_t] * 3 + [Config.unknown_edge_sigma_r] * 3) ** 2)


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
            augment: Optional[TransformerSolverConfig] = None,
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
        self.augment = augment   # its depth_* / odom_* / world_* fields; None for val

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
            odometry_aug = None
            if self.augment is not None:
                data = dict(data)
                descriptors, odometry_aug = augment(data, keep, descriptors, self.augment)
            if any(self.pose_noise):
                data = dict(data)   # the input poses change, so do the targets model_item derives from them
                data['poses'] = perturb_poses(data['poses'], *self.pose_noise)
            item = model_item(data, keep, descriptors, self.closure_dropout)
            if odometry_aug is not None:   # not in MODEL_INPUTS: a label, never seen by the model
                item['odometry_aug'] = odometry_aug
            return item


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


# Load-time augmentation (TransformerSolverConfig's depth_* / odom_* / world_*). torch's RNG throughout, as
# perturb_poses, and nothing drawn by a step that is off: with everything off an item is bit-identical.

def _randn(*shape) -> np.ndarray:
    return torch.randn(shape, dtype=torch.float64).numpy()


def _rand(*shape) -> np.ndarray:
    return torch.rand(shape, dtype=torch.float64).numpy()


def _random_vectors(n: int, lo: float, hi: float) -> np.ndarray:
    """(n, 3): uniform directions (normalised Gaussian), lengths U(lo, hi)."""
    v = _randn(n, 3)
    return v / np.linalg.norm(v, axis=1, keepdims=True) * (lo + (hi - lo) * _rand(n, 1))


def _transforms(rotvec: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(n, 4, 4) [Exp(rotvec) | t]: SO(3) x R^3, as delta_targets encodes, not the SE(3) Exp."""
    T = np.tile(np.eye(4), (len(rotvec), 1, 1))
    T[:, :3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    T[:, :3, 3] = t
    return T


def augment_points(points: np.ndarray, descriptors: Optional[Tensor], cfg: TransformerSolverConfig
                   ) -> tuple[np.ndarray, Optional[Tensor]]:
    """
    (V, P, 3) camera-frame points through a depth-sensor model, same shape and dtype; descriptors
    (V, P, D) rows follow their point. Holes first (copies, so a copy still gets its own noise), then
    z' = z + N(0, s0 + s1 z^2), then flying pixels. Along the ray: p' = p z'/z, z the planar depth.
    z <= 0 is sample_points' empty keyframe: left alone, never a copy source.
    """
    p = points.astype(np.float64)
    V, P, _ = p.shape
    valid = p[..., 2] > 0
    if cfg.depth_hole_fraction:
        hole = (_rand(V, P) < cfg.depth_hole_fraction) & valid
        for v in np.nonzero(hole.any(axis=1))[0]:
            source = np.nonzero(valid[v] & ~hole[v])[0]
            if len(source) == 0:
                continue
            dst = np.nonzero(hole[v])[0]
            src = source[torch.randint(len(source), (len(dst),)).numpy()]
            p[v, dst] = p[v, src]
            if descriptors is not None:
                descriptors[v, torch.from_numpy(dst)] = descriptors[v, torch.from_numpy(src)]
    z = np.where(valid, p[..., 2], 1.0)
    z_new = z.copy()
    if cfg.depth_noise_s0 or cfg.depth_noise_s1:
        # floor 1 cm: never through the camera (sigma at 5 m is 5.5 cm with the suggested values)
        z_new = np.maximum(z + (cfg.depth_noise_s0 + cfg.depth_noise_s1 * z ** 2) * _randn(V, P), 0.01)
    if cfg.depth_flying_fraction:
        fly = _rand(V, P) < cfg.depth_flying_fraction
        z_max = np.maximum(np.where(valid, z, 0).max(axis=1, keepdims=True), 0.2)
        z_new = np.where(fly, 0.2 + (z_max - 0.2) * _rand(V, P), z_new)
    p *= np.where(valid, z_new / z, 1.0)[..., None]
    return p.astype(points.dtype), descriptors


def augment_odometry(Z: np.ndarray, info: np.ndarray, cfg: TransformerSolverConfig
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    (O, 4, 4) odometry Z and (O, 6, 6) information after noise, break and dropout, plus (O,) int8 per edge:
    0 untouched by break/dropout, 1 broken, 2 hidden. Dropout's constant-velocity guess is the previous
    edge's Z as already augmented, so a run of hidden edges repeats one motion.
    """
    Z, info = Z.copy(), info.copy()
    O = len(Z)
    code = np.zeros(O, dtype=np.int8)
    if cfg.odom_noise_deg or cfg.odom_noise_m:
        Z = Z @ _transforms(_randn(O, 3) * np.radians(cfg.odom_noise_deg), _randn(O, 3) * cfg.odom_noise_m)
    if cfg.odom_outlier_prob:
        hit = _rand(O) < cfg.odom_outlier_prob
        n = int(hit.sum())
        Z[hit] = Z[hit] @ _transforms(
            _random_vectors(n, np.radians(cfg.odom_outlier_deg_min), np.radians(cfg.odom_outlier_deg_max)),
            _random_vectors(n, cfg.odom_outlier_m_min, cfg.odom_outlier_m_max))
    if cfg.odom_break_prob and O > 1 and _rand() < cfg.odom_break_prob:
        e = 1 + int(torch.randint(O - 1, ()))
        Z[e] = Z[e] @ _transforms(
            _random_vectors(1, np.radians(cfg.odom_break_deg_min), np.radians(cfg.odom_break_deg_max)),
            _random_vectors(1, 0.0, cfg.odom_break_m))[0]
        info[e], code[e] = UNKNOWN_INFO, 1
    if cfg.odom_dropout:
        hidden = _rand(O) < cfg.odom_dropout
        hidden[0] = False
        for e in np.nonzero(hidden)[0]:
            Z[e], info[e], code[e] = Z[e - 1], UNKNOWN_INFO, 2
    return Z, info, code


def chain_poses(poses: np.ndarray, ij: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """(V, 4, 4) T_j = T_i Z_ij in edge order, vertex 0 (and any vertex no edge reaches) as given."""
    T = poses.copy()
    for (i, j), z in zip(ij, Z):
        T[j] = T[i] @ z
    return T


def world_transform(cfg: TransformerSolverConfig) -> np.ndarray:
    """(4, 4) G for world_rotation / world_translation_m."""
    if cfg.world_rotation == "so3":
        q = _randn(4)   # a normalised 4-D Gaussian is a uniform unit quaternion: Haar on SO(3)
        R = Rotation.from_quat(q / np.linalg.norm(q)).as_matrix()
    elif cfg.world_rotation == "yaw":
        R = Rotation.from_rotvec([0.0, 2 * np.pi * _rand(), 0.0]).as_matrix()
    elif cfg.world_rotation == "none":
        R = np.eye(3)
    else:
        raise ValueError(f"world_rotation {cfg.world_rotation!r}: none | yaw | so3")
    G = np.eye(4)
    G[:3, :3] = R
    if cfg.world_translation_m:
        G[:3, 3] = cfg.world_translation_m * (2 * _rand(3) - 1)
    return G


def augment(data: dict, keep: slice, descriptors: Optional[Tensor], cfg: TransformerSolverConfig
            ) -> tuple[Optional[Tensor], Optional[np.ndarray]]:
    """
    data (sample.npz as a dict) augmented in place for model_item: points (the kept window only), odometry,
    poses re-chained from the augmented odometry, then G on poses and gt_poses. Closures untouched. Returns
    the descriptors (rows moved with the hole copies) and, when break or dropout is on, the (O,) odometry_aug
    codes as a tensor. Pose noise (perturb_poses) comes after, on the re-chained poses: G T Exp = G (T Exp).
    """
    if cfg.depth_noise_s0 or cfg.depth_noise_s1 or cfg.depth_hole_fraction or cfg.depth_flying_fraction:
        points = data['points'].copy()
        points[:, keep], descriptors = augment_points(points[:, keep], descriptors, cfg)
        data['points'] = points
    code = None
    if cfg.odom_noise_deg or cfg.odom_noise_m or cfg.odom_outlier_prob or cfg.odom_break_prob or cfg.odom_dropout:
        data['odometry_Z'], data['odometry_info'], code = augment_odometry(data['odometry_Z'], data['odometry_info'], cfg)
        data['poses'] = chain_poses(data['poses'], data['odometry_ij'], data['odometry_Z'])
    if cfg.world_rotation != "none" or cfg.world_translation_m:
        G = world_transform(cfg)
        data['poses'], data['gt_poses'] = G @ data['poses'], G @ data['gt_poses']
    labelled = cfg.odom_break_prob or cfg.odom_dropout
    return descriptors, (torch.from_numpy(code) if labelled else None)
