"""
Classical Gauss-Newton vs the Phase 3 transformer, on one fixed subset of val
trajectories and one ATE. The GN side is slow (rebuilds every graph through the
frontend), so experiments/gn_baseline.py runs it once and caches it to
baseline_path(n); the transformer side runs here, from training or by hand.

Both are scored on the graph's VERTICES in the vertex-0 frame: vertex k is
sequence frame vertex_frames[k], not frame k, once a keyframe is bridged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from visual_pose import _geometry as cpp
from visual_pose.config.config import TransformerSolverConfig
from visual_pose.config.constants import REPO_DIR
from visual_pose.data_utils.loaders import collate_pose_graph_ds, split_trajectories
from visual_pose.data_utils.pose_graph_dataset import PoseGraphDataset
from visual_pose.data_utils.pose_graph_samples import SAMPLE_FILE
from visual_pose.evaluation import metrics


def eval_subset(n: int) -> list[Path]:
    """The first n val trajectories of split_trajectories(), path-sorted. The ONE definition of the subset."""
    return sorted(split_trajectories()[1])[:n]


def trajectory_key(traj_dir: Path) -> str:
    return f"{traj_dir.parents[1].name}/{traj_dir.name}"


def baseline_path(n: int) -> Path:
    return REPO_DIR / "out" / f"gn_baseline_val{n}.json"


def ates(T: np.ndarray, T_gt: np.ndarray) -> dict[str, float]:
    """Anchored and aligned ATE of (V, 4, 4) poses, both already in the vertex-0 frame."""
    p, g = T[:, :3, 3], T_gt[:, :3, 3]
    return {"ate": float(metrics.ate(p, g)), "ate_aligned": float(metrics.ate_aligned(p, g))}


def summarise(per: dict[str, dict], fields: list[str]) -> dict[str, dict[str, float]]:
    return {f: {"mean": float(np.mean([r[f] for r in per.values()])),
                "median": float(np.median([r[f] for r in per.values()])),
                "std": float(np.std([r[f] for r in per.values()]))} for f in fields}


def correction(delta: np.ndarray) -> np.ndarray:
    """
    (4, 4) correction from one vertex's [rho, phi]: rotation Exp(phi), translation rho
    directly -- SO(3) x R^3, as chordial_distance trains it against T_in^-1 T_gt. Not the
    SE(3) Exp, whose translation V(phi) rho differs once phi is large (the flipped graphs).
    """
    T = np.eye(4)
    T[:3, :3] = cpp.RotationSO3.exp(delta[3:].reshape(3, 1)).matrix()
    T[:3, 3] = delta[:3]
    return T


@torch.no_grad()
def refine(model: nn.Module, batch: dict, T_in: np.ndarray, cfg: TransformerSolverConfig) -> tuple[np.ndarray, float]:
    """
    (V,4,4) poses after cfg.refinement_iterations passes, as trans_step: T_i <- T_i correction(delta_i)
    (right perturbation), the model re-run on the updated poses. batch is one collated graph on the
    model's device; its poses are overwritten. Also the largest |delta| seen. Shared by the ATE
    evaluation and the live solver (pose_graph/solvers.py).
    """
    from visual_pose.training.transformer_solver import model_input
    V = len(T_in)
    T, delta_max = T_in, 0.0
    for _ in range(cfg.refinement_iterations):
        delta = model(**model_input(batch, cfg.rotate_points))[0, :V].cpu().double().numpy()
        T = np.stack([T[v] @ correction(delta[v]) for v in range(V)])
        # model sees the float32 of the float64 T; trans_step compounds in float32 instead
        batch["poses"][0, :V] = torch.from_numpy(T[:, :3].reshape(V, 12)).to(batch["poses"])
        delta_max = max(delta_max, float(np.abs(delta).max()))
    return T, delta_max


@torch.no_grad()
def transformer_ate(model: nn.Module, trajectories: list[Path], cfg: TransformerSolverConfig,
                    device: torch.device) -> dict:
    """
    Per trajectory, refine()'s poses scored against gt. T_in and gt come from sample.npz in
    float64, not the dataset's float32 features, so delta = 0 reproduces the odometry ATE exactly.
    """
    # closures as build_transformer_loaders' val set: none for a closure_dropout = 1.0 model,
    # which never trained its closure embedding, all of them otherwise
    ds = PoseGraphDataset(trajectories, pass_descriptor=cfg.use_descriptor,
                          num_points=cfg.points_per_keyframe, random_points=False,
                          closure_dropout=1.0 if cfg.closure_dropout == 1.0 else 0.0)
    was_training = model.training
    model.eval()
    per = {}
    try:
        for i, traj in enumerate(trajectories):
            batch = collate_pose_graph_ds([ds[i]])
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with np.load(traj / SAMPLE_FILE) as data:
                T_in, T_gt = data["poses"], data["gt_poses"]
            T, delta_max = refine(model, batch, T_in, cfg)
            per[trajectory_key(traj)] = {"V": len(T_in), **ates(T, T_gt), "delta_max": delta_max}
    finally:
        model.train(was_training)
    return {"per_trajectory": per, **summarise(per, ["ate", "ate_aligned"])}


def compare_line(result: dict, baseline: dict | None) -> str:
    """One line: transformer mean/median/std ATE against the cached odometry and GN."""
    fmt = lambda a: f"{a['mean']:.3f} / {a['median']:.3f} ± {a['std']:.3f}"   # mean / median ± std over graphs
    t = f"transformer ATE {fmt(result['ate'])} m"
    if baseline is None:
        return t
    # recomputed from the per-graph results: older caches' summaries have no std
    s = summarise(baseline["per_trajectory"], ["odom_ate", "odom_ate_aligned", "gn_ate", "gn_ate_aligned"])
    return (f"{t} | odometry {fmt(s['odom_ate'])} | GN {fmt(s['gn_ate'])}"
            f"  (aligned: transformer {fmt(result['ate_aligned'])},"
            f" odometry {fmt(s['odom_ate_aligned'])}, GN {fmt(s['gn_ate_aligned'])})")


def ate_epoch_hook(model: nn.Module, cfg: TransformerSolverConfig):
    """fit's on_epoch_end(epoch, value): every cfg.ate_eval_every epochs, transformer vs cached GN."""
    trajectories = eval_subset(cfg.ate_eval_trajectories)
    path = baseline_path(cfg.ate_eval_trajectories)
    baseline = json.loads(path.read_text()) if path.exists() else None
    if baseline is None:
        print(f"[ate] no GN baseline at {path}: run "
              f"`PYTHONPATH=python:python/visual_pose:. .venv/bin/python experiments/gn_baseline.py "
              f"--n {cfg.ate_eval_trajectories}`; reporting the transformer alone")
    elif baseline["trajectories"] != [trajectory_key(t) for t in trajectories]:
        print(f"[ate] WARNING: {path} was computed on a different subset (HM3D_SCENES changed?); "
              "its numbers are not comparable")

    def hook(epoch: int, value: float) -> None:
        if (epoch + 1) % cfg.ate_eval_every:
            return
        device = next(model.parameters()).device
        result = transformer_ate(model, trajectories, cfg, device)
        print(f"[ate] epoch {epoch + 1}, {len(trajectories)} val graphs: {compare_line(result, baseline)}")
    return hook
