"""
Numbers about an estimate: per-edge error, and the trajectory it chains into.

Chaining lives here rather than in its own module because every metric below
consumes its output -- an (N+1, 3) path is the unit the trajectory numbers are
computed on.
"""
from __future__ import annotations

import numpy as np


def pose_error(T_est: np.ndarray | None,
               T_gt: np.ndarray) -> tuple[float, float, float]:
    """(rotation deg, translation direction deg, |t| error cm) for one edge."""
    if T_est is None:
        return np.nan, np.nan, np.nan
    dR = T_est[:3, :3] @ T_gt[:3, :3].T
    rot = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    t, t_gt = T_est[:3, 3], T_gt[:3, 3]
    cos = abs(t @ t_gt) / (np.linalg.norm(t) * np.linalg.norm(t_gt))
    return (rot, np.degrees(np.arccos(np.clip(cos, -1, 1))),
            100 * abs(np.linalg.norm(t) - np.linalg.norm(t_gt)))


def chain(relative: list[np.ndarray | None], gt: list[np.ndarray],
          metric: bool) -> np.ndarray:
    """Fold relative poses into (N+1, 3) world positions. Positions only."""
    T = np.eye(4)
    out = [T[:3, 3].copy()]
    for T_est, T_gt in zip(relative, gt):
        if T_est is None:
            out.append(out[-1])
            continue
        T_edge = T_est.copy()
        if not metric:
            t = T_est[:3, 3]
            T_edge[:3, 3] = t / np.linalg.norm(t) * np.linalg.norm(T_gt[:3, 3])
        T = T @ np.linalg.inv(T_edge)      # camera->world accumulates the inverse
        out.append(T[:3, 3].copy())
    return np.array(out)


def path_length(path: np.ndarray) -> float:
    """Total path length in metres; the denominator for drift."""
    return np.linalg.norm(np.diff(path, axis=0), axis=1).sum()


def ate(path: np.ndarray, gt_path: np.ndarray) -> float:
    """RMS position error against ground truth, metres. No alignment: the
    poses are metric, so a fitted scale would hide real error.
    """
    pass


def drift(path: np.ndarray, gt_path: np.ndarray) -> tuple[float, float]:
    """Endpoint error, in metres and as a fraction of path length.

    Both, because a 1 m miss means different things over 10 m and 100 m.
    """
    final = float(np.linalg.norm(path[-1] - gt_path[-1]))
    return final, 100 * final / path_length(gt_path)
