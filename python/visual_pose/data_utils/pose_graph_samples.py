"""
Phase 3 samples: each trajectory's graph.g2o + graph.npz + depth, reduced once to
sample.npz beside them, so training reads ~1.5 MB per graph and never a frame.

    PYTHONPATH=python .venv/bin/python -m visual_pose.data_utils.pose_graph_samples [--overwrite]

sample.npz (V vertices, L closures, O odometry edges, P points), full matrices;
PoseGraphDataset trims them to its (V, 12) / (E, 33) features:
    poses          (V,4,4)  the unsolved graph's vertices: chained odometry, vertex 0 = identity
    gt_poses       (V,4,4)  ground truth re-anchored to vertex 0: inv(gt_0) @ gt_i. TARGET ONLY
    points         (V,P,3)  depth points, camera frame (OpenCV), metres
    closures_ij    (L,2)    vertex indices; Z_ij = T_ij = inv(T_i) @ T_j (WorldModel.measurement)
    closures_Z     (L,4,4)
    closure_info   (L,6,6)  raw, uncalibrated: no cluster 1/size, no median rescale
    closure_rot_err_deg, closure_t_err_m   (L,)  Z_ij against inv(gt_i) @ gt_j
    odometry_ij, odometry_Z, odometry_info  as the closure arrays, (O, ...)
    vertex_frames  (V,)     sequence index of each vertex
    split                   "train" | "val", from graph.npz
"""
import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import nn, Tensor

from visual_pose.data_utils.constants import DEVICE
from visual_pose.matching import bilinear_sample_descriptor
from visual_pose.models.dino_v2 import SemanticDescriptor, PATCH
from visual_pose.config import TransformerSolverConfig
from visual_pose.geometry.true_correspondences import back_project
from visual_pose.data_utils.hm3d_sequence import POSES_FILE, scene_dirs, HM3DSequence
from visual_pose.data_utils.sources import SavedHM3DFrames
from visual_pose.pose_graph.frontend import GRAPH_FILE, GRAPH_SIDECAR

SAMPLE_FILE = "sample.npz"
_UPPER = np.triu_indices(6)
CFG = TransformerSolverConfig()
DESCRIPTOR_BATCH_SIZE = 128


def _pose(xyz_quat: list[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(xyz_quat[3:7]).as_matrix()   # g2o order: qx qy qz qw, as scipy
    T[:3, 3] = xyz_quat[:3]
    return T


def read_g2o(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(poses (V,4,4), edge_ij (E,2), Z (E,4,4), info (E,6,6)), edges in file order."""
    poses, ij, Z, info = [], [], [], []
    for line in path.read_text().splitlines():
        tag, *rest = line.split()
        if tag == "VERTEX_SE3:QUAT":
            poses.append(_pose([float(x) for x in rest[1:8]]))
        elif tag == "EDGE_SE3:QUAT":
            ij.append((int(rest[0]), int(rest[1])))
            Z.append(_pose([float(x) for x in rest[2:9]]))
            I = np.zeros((6, 6))
            I[_UPPER] = [float(x) for x in rest[9:30]]
            info.append(I + np.triu(I, 1).T)
    return np.array(poses), np.array(ij).reshape(-1, 2), np.array(Z).reshape(-1, 4, 4), np.array(info).reshape(-1, 6, 6)


def sample_points(frames: SavedHM3DFrames, frame: int, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """n points backprojected from frame's depth, uniform over pixels with a return; (n, 3)."""
    depth = frames.depth(frame)
    v, u = np.nonzero(depth > 0)
    if len(v) == 0:
        return np.zeros((n, 3), np.float32), np.zeros((n, 2), np.float32)
    pick = rng.choice(len(v), n, replace=len(v) < n)
    v, u, z = v[pick], u[pick], depth[v[pick], u[pick]]
    K = frames.K
    uvs = np.stack([u, v], axis=1)
    points = back_project(uvs, z, K)
    return points, uvs


def relative_errors(ij: np.ndarray, Z: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation (deg) and translation (m) error of each Z_ij against inv(gt_i) @ gt_j."""
    T_true = np.linalg.inv(gt[ij[:, 0]]) @ gt[ij[:, 1]]
    E = np.linalg.inv(T_true) @ Z
    cos = np.clip((np.trace(E[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cos)), np.linalg.norm(E[:, :3, 3], axis=1)


def build_sample(traj_dir: Path, points: int = CFG.num_point_samples) -> Path:
    side = np.load(traj_dir / GRAPH_SIDECAR)
    poses_in, ij, Z, info = read_g2o(traj_dir / GRAPH_FILE)
    vertex_frames, n_odom = side["vertex_frames"], int(side["n_odometry"])
    gt = side["gt_poses"]
    gt = np.linalg.inv(gt[0]) @ gt
    frames = SavedHM3DFrames(traj_dir)
    rng = np.random.default_rng(0)
    pts, uvs = [], []
    for f in vertex_frames:
        pt, uv = sample_points(frames, int(f), points, rng)
        pts.append(pt)
        uvs.append(uv)
    pts, uvs = np.stack(pts, dtype=np.float32), np.stack(uvs, dtype=np.int64)
    rot_err, t_err = relative_errors(ij[n_odom:], Z[n_odom:], gt)
    out = traj_dir / SAMPLE_FILE
    tmp = out.with_name(f".{SAMPLE_FILE}")
    with open(tmp, "wb") as f:
        np.savez(f, poses=poses_in, gt_poses=gt, points=pts, uvs=uvs,
                 closures_ij=ij[n_odom:], closures_Z=Z[n_odom:], closure_info=info[n_odom:],
                 closure_rot_err_deg=rot_err, closure_t_err_m=t_err,
                 odometry_ij=ij[:n_odom], odometry_Z=Z[:n_odom], odometry_info=info[:n_odom],
                 vertex_frames=vertex_frames, split=side["split"] if "split" in side else np.array("None"))
    tmp.replace(out)
    return out


def is_current(traj_dir: Path) -> bool:
    """sample.npz exists and is newer than the graph it was made from."""
    sample = traj_dir / SAMPLE_FILE
    return sample.exists() and sample.stat().st_mtime >= (traj_dir / GRAPH_FILE).stat().st_mtime


def _build(traj_dir: Path) -> str:
    try:
        build_sample(traj_dir)
        return f"{traj_dir.parents[1].name}/{traj_dir.name}"
    except Exception as e:     # one bad trajectory should not cost the other 999
        return f"FAILED {traj_dir}: {e!r}"


def graph_dirs() -> list[Path]:
    return [d for s in scene_dirs() for d in sorted((s / "trajectories").glob("trajectory_*"))
            if (d / POSES_FILE).exists() and (d / GRAPH_FILE).exists()]


def save_descriptors(descriptors: torch.Tensor, traj: Path) -> None:
    """Adds descriptors (V, P, D) float16 to traj's sample.npz, rewriting it."""
    with np.load(traj / SAMPLE_FILE) as data:
        data_dict = dict(data)
    data_dict['descriptors'] = descriptors.cpu().numpy().astype(np.float16)
    np.savez(traj / SAMPLE_FILE, **data_dict)

def save_descriptors_at(seq: SavedHM3DFrames, descriptor_model, uvs: Tensor,
                     batch_size: int = 128) -> None:
    """descriptor_model's map read at each stored point's pixel uvs (V, P, 2): (V, P, D) into sample.npz."""
    with np.load(seq.dir / GRAPH_SIDECAR) as side:
        kfs = side["vertex_frames"]
    rgb = np.stack([seq.rgb(int(i)) for i in kfs])          # (N, H, W, 3) uint8: 1 GB, not 4
    H, W = rgb.shape[-3:-1]
    h, w = H - H % PATCH, W - W % PATCH
    top, left = (H - h) // 2, (W - w) // 2
    descriptors = []
    with torch.inference_mode():
        for b in range(0, len(rgb), batch_size):
            x = torch.from_numpy(rgb[b:b + batch_size]).to(DEVICE).permute(0, 3, 1, 2).float().div_(255)
            raw_descriptors = descriptor_model(x)
            batch_uvs = uvs[b:b + batch_size]
            feature_uvs = torch.stack([
                ((batch_uvs[..., 0] - left) + 0.5) * (raw_descriptors.shape[3] / w) - 0.5,
                ((batch_uvs[..., 1] - top) + 0.5) * (raw_descriptors.shape[2] / h) - 0.5,
            ], dim=-1)
            descriptors.append(bilinear_sample_descriptor(raw_descriptors, feature_uvs).cpu())
    save_descriptors(torch.concat(descriptors), seq.dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true", help="rebuild samples that are already current")
    args = ap.parse_args()
    todo = [d for d in graph_dirs() if args.overwrite or not is_current(d)]
    print(f"{len(todo)} samples to build", flush=True)
    with Pool(max(1, cpu_count() - 2)) as pool:
        for n, msg in enumerate(pool.imap_unordered(_build, todo), 1):
            if msg.startswith("FAILED") or n % 50 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} {msg}", flush=True)
    descriptor_model = SemanticDescriptor()
    trajs = graph_dirs()
    for n, traj in enumerate(trajs, 1):
        if n % 50 == 0 or n == len(trajs):
            print(f"descriptors {n}/{len(trajs)}", flush=True)
        try:
            seq = SavedHM3DFrames(traj)
            with np.load(traj / SAMPLE_FILE) as data:
                if len(data.get('descriptors', [])) == len(data['points']) and not args.overwrite:
                    continue
                uvs = torch.from_numpy(data['uvs']).int().to(DEVICE)
                save_descriptors_at(seq, descriptor_model, uvs)
                torch.accelerator.empty_cache()
        except Exception as e:
            print(f"FAILED {traj}: {e}", flush=True)






