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

Beside it, from the descriptor pass (second half of __main__), float16:
    descriptors_pvd.npy  (P,V,D)  DINOv2 patch descriptors read at each stored point. POINT-MAJOR
    cls.npy              (V,D)    each keyframe's DINOv2 CLS token
Raw .npy, not inside sample.npz: PoseGraphDataset memory-maps the descriptors and reads
only the window of points it keeps (an .npz member can only be read whole: 120 MB a graph),
and adding them never rewrites sample.npz. Point-major so that window, across every
keyframe, is ONE contiguous block: on the USB hard disk (Seagate Portable, ~10 ms a seek)
(V,P,D) costs a seek per keyframe, ~300 a graph, ~3 s -- slower than reading it all.
"""
import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import torch
from torch import nn, Tensor

from visual_pose import _geometry as cpp
from visual_pose.config.constants import DEVICE
from visual_pose.geometry.matching import bilinear_sample_descriptor
from visual_pose.models.dino_v2 import SemanticDescriptor, PATCH
from visual_pose.config.config import TransformerSolverConfig
from visual_pose.geometry.true_correspondences import back_project
from visual_pose.data_utils.hm3d_sequence import POSES_FILE, scene_dirs, HM3DSequence
from visual_pose.data_utils.sources import SavedHM3DFrames
from visual_pose.pose_graph.frontend import GRAPH_FILE, GRAPH_SIDECAR

SAMPLE_FILE = "sample.npz"
DESCRIPTOR_FILE = "descriptors_pvd.npy"
CLS_FILE = "cls.npy"
OLD_DESCRIPTOR_FILE = "descriptors.npy"   # (V,P,D), briefly written 2026-10-15; split_out_descriptors converts it
CFG = TransformerSolverConfig()
DESCRIPTOR_BATCH_SIZE = 128


def graph_arrays(graph: cpp.FactorGraph) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    (poses (V,4,4), edge_ij (E,2), Z (E,4,4), info (E,6,6)), edges in insertion order. A graph in
    memory (the live solver) or loaded from graph.g2o: on 230 saved graphs the loader matches the
    old text parser to 1e-13 (poses) and exactly (info).
    """
    edges = graph.edges
    return (np.array(graph.poses()).reshape(-1, 4, 4),
            np.array([(e.from_index, e.to_index) for e in edges], dtype=np.int64).reshape(-1, 2),
            np.array([e.measured_pose.matrix() for e in edges]).reshape(-1, 4, 4),
            np.array([e.info_matrix for e in edges]).reshape(-1, 6, 6))


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


def sample_arrays(graph: cpp.FactorGraph, edge_is_closure: np.ndarray, vertex_frames: np.ndarray, frames,
                  points: int = CFG.num_point_samples, gt: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """
    The sample.npz arrays (module docstring) for one unsolved graph; the training samples and the
    live transformer solver both come through here. frames: anything with depth(frame) and K.
    gt (V,4,4) adds the TARGET-ONLY arrays; without it (a live run) they are absent.
    """
    poses_in, ij, Z, info = graph_arrays(graph)
    closure = np.asarray(edge_is_closure, dtype=bool)
    rng = np.random.default_rng(0)   # same seed, same vertex order -> the same points every rebuild
    pts, uvs = zip(*(sample_points(frames, int(f), points, rng) for f in vertex_frames))
    data = dict(poses=poses_in, points=np.stack(pts, dtype=np.float32), uvs=np.stack(uvs, dtype=np.int64),
                closures_ij=ij[closure], closures_Z=Z[closure], closure_info=info[closure],
                odometry_ij=ij[~closure], odometry_Z=Z[~closure], odometry_info=info[~closure],
                vertex_frames=np.asarray(vertex_frames))
    if gt is not None:
        data["gt_poses"] = np.linalg.inv(gt[0]) @ gt
        data["closure_rot_err_deg"], data["closure_t_err_m"] = relative_errors(ij[closure], Z[closure], data["gt_poses"])
    return data


def build_sample(traj_dir: Path, points: int = CFG.num_point_samples) -> Path:
    with np.load(traj_dir / GRAPH_SIDECAR) as side:
        vertex_frames, n_odom = side["vertex_frames"], int(side["n_odometry"])
        gt, split = side["gt_poses"], side["split"] if "split" in side else np.array("None")
    graph = cpp.FactorGraph(str(traj_dir / GRAPH_FILE))
    data = sample_arrays(graph, np.arange(len(graph.edges)) >= n_odom, vertex_frames,
                         SavedHM3DFrames(traj_dir), points, gt)
    uvs = data["uvs"]
    out = traj_dir / SAMPLE_FILE
    # Descriptors are read at uvs in each keyframe's rgb, so they outlive a new graph unless the
    # keyframes moved (frontend bridging) -- rng(0) in vertex_frames order redraws the same uvs.
    same_points = False
    if out.exists():
        with np.load(out) as old:
            same_points = (np.array_equal(old["vertex_frames"], vertex_frames)
                           and np.array_equal(old["uvs"], uvs))
    tmp = out.with_name(f".{SAMPLE_FILE}")
    with open(tmp, "wb") as f:
        np.savez(f, **data, split=split)
    tmp.replace(out)
    if not same_points:
        for name in (DESCRIPTOR_FILE, OLD_DESCRIPTOR_FILE, CLS_FILE):   # sampled at the old points
            (traj_dir / name).unlink(missing_ok=True)
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


def _save_npy(path: Path, array: np.ndarray) -> None:
    """Via a temp name and rename: an interrupted write leaves the old file or none, never half of one."""
    tmp = path.with_name(f".{path.name}")
    with open(tmp, "wb") as f:
        np.save(f, array)
    tmp.replace(path)


def save_descriptors(descriptors: np.ndarray, cls: np.ndarray, traj: Path) -> None:
    """descriptors (V, P, D), stored transposed to (P, V, D), and cls (V, D), float16, beside sample.npz."""
    _save_npy(traj / CLS_FILE, cls.astype(np.float16))
    _save_npy(traj / DESCRIPTOR_FILE, np.ascontiguousarray(descriptors.transpose(1, 0, 2)).astype(np.float16))


def split_out_descriptors(traj: Path) -> None:
    """
    Old layouts -> new, no GPU: descriptors and cls inside sample.npz, or a (V,P,D)
    OLD_DESCRIPTOR_FILE, become DESCRIPTOR_FILE / CLS_FILE. Descriptors without cls are
    only dropped -- the descriptor pass recomputes both.
    """
    old = traj / OLD_DESCRIPTOR_FILE
    if old.exists():
        if (traj / CLS_FILE).exists():
            save_descriptors(np.load(old), np.load(traj / CLS_FILE), traj)
        old.unlink()
    with np.load(traj / SAMPLE_FILE) as data:
        if 'descriptors' not in data:
            return
        if 'cls' in data:
            save_descriptors(data['descriptors'], data['cls'], traj)
        rest = {k: data[k] for k in data.files if k not in ('descriptors', 'cls')}
    out = traj / SAMPLE_FILE
    tmp = out.with_name(f".{SAMPLE_FILE}")
    with open(tmp, "wb") as f:
        np.savez(f, **rest)
    tmp.replace(out)


def descriptors_at(frames, vertex_frames: np.ndarray, descriptor_model, uvs: Tensor,
                   batch_size: int = DESCRIPTOR_BATCH_SIZE) -> tuple[Tensor, Tensor]:
    """
    descriptor_model's map read at pixels uvs (V, P, 2) of each vertex's rgb: (V, P, D), and each
    keyframe's CLS (V, D), float32 on the CPU. The descriptor pass and the live transformer
    solver both come through here. frames: anything with rgb(frame).
    """
    rgb = np.stack([frames.rgb(int(i)) for i in vertex_frames])          # (V, H, W, 3) uint8: 1 GB, not 4
    return descriptor_model.descriptors_for_images(torch.from_numpy(rgb), uvs, batch_size)


def save_descriptors_at(seq: SavedHM3DFrames, descriptor_model, uvs: Tensor,
                     batch_size: int = DESCRIPTOR_BATCH_SIZE) -> None:
    """descriptors_at every stored point (uvs (V, P, 2)), saved by save_descriptors."""
    with np.load(seq.dir / GRAPH_SIDECAR) as side:
        kfs = side["vertex_frames"]
    descriptors, cls = descriptors_at(seq, kfs, descriptor_model, uvs, batch_size)
    save_descriptors(descriptors.numpy(), cls.numpy(), seq.dir)


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
            split_out_descriptors(traj)
            if (traj / DESCRIPTOR_FILE).exists() and (traj / CLS_FILE).exists() and not args.overwrite:
                continue
            with np.load(traj / SAMPLE_FILE) as data:
                uvs = torch.from_numpy(data['uvs']).int().to(DEVICE)
            save_descriptors_at(SavedHM3DFrames(traj), descriptor_model, uvs)
            torch.accelerator.empty_cache()
        except Exception as e:
            print(f"FAILED {traj}: {e}", flush=True)






