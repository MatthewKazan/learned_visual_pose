"""
Numbers about an estimate: per-edge error, and the trajectory it chains into.

Chaining lives here rather than in its own module because every metric below
consumes its output -- an (N+1, 3) path is the unit the trajectory numbers are
computed on.
"""
from __future__ import annotations

import numpy as np

from visual_pose import _geometry as cpp


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


def chain_poses(relative: list[np.ndarray | None], gt: list[np.ndarray],
                metric: bool) -> np.ndarray:
    """Fold relative poses into (N+1, 4, 4) T_0k: camera k into the first
    camera's frame. Rotation kept, for the point-cloud viewer."""
    T = np.eye(4)
    out = [T.copy()]
    for T_est, T_gt in zip(relative, gt):
        if T_est is None:
            out.append(out[-1])
            continue
        T_edge = T_est.copy()
        if not metric:
            t = T_est[:3, 3]
            T_edge[:3, 3] = t / np.linalg.norm(t) * np.linalg.norm(T_gt[:3, 3])
        T = T @ np.linalg.inv(T_edge)      # camera->world accumulates the inverse
        out.append(T.copy())
    return np.array(out)


def chain(relative: list[np.ndarray | None], gt: list[np.ndarray],
          metric: bool) -> np.ndarray:
    """Fold relative poses into (N+1, 3) world positions. Positions only."""
    return chain_poses(relative, gt, metric)[:, :3, 3]


def path_length(path: np.ndarray) -> float:
    """Total path length in metres; the denominator for drift."""
    return np.linalg.norm(np.diff(path, axis=0), axis=1).sum()


def ate(path: np.ndarray, gt_path: np.ndarray) -> float:
    """RMS position error against ground truth, metres, both expressed in
    the frame of the first pose. No alignment, so this also charges the
    rotation error of the first edge, times the lever arm: on P000 a 0.23 deg
    error there is 0.84 m here and 0.15 m after ate_aligned (2026-09-29).
    """
    assert len(path) == len(gt_path)
    sum_e = 0.0
    # ignore the first point, it's always the same as ground truth
    for i in range(1, len(path)):
        sum_e += np.linalg.norm(path[i] - gt_path[i]) ** 2

    return np.sqrt(sum_e / (len(path) - 1))



def ate_aligned(path: np.ndarray, gt_path: np.ndarray) -> float:
    """
    RMS position error after the best RIGID alignment of the estimate onto
    ground truth: rotation and translation, no scale. Removes only the gauge
    (which frame the anchor happens to define), never metric error -- a fitted
    scale would, and is deliberately not fitted. The standard ATE definition.
    The alignment is the same Kabsch fit the frontend uses on point clouds.
    """
    T = rigid_alignment(path, gt_path)
    aligned = path @ T[:3, :3].T + T[:3, 3]
    return float(np.sqrt(((aligned - gt_path) ** 2).sum(axis=1).mean()))


def rigid_alignment(path: np.ndarray, gt_path: np.ndarray) -> np.ndarray:
    """The (4, 4) ate_aligned fits: gt ~ R path + t. Separate so the viewers
    can draw an estimate the way ate_aligned scores it."""
    assert len(path) == len(gt_path)
    return cpp.kabsch_algorithm(np.asarray(path, dtype=np.float64),
                                np.asarray(gt_path, dtype=np.float64)).T_ji.matrix()


def drift(path: np.ndarray, gt_path: np.ndarray) -> tuple[float, float]:
    """Endpoint error, in metres and as a fraction of path length.

    Both, because a 1 m miss means different things over 10 m and 100 m.
    """
    final = float(np.linalg.norm(path[-1] - gt_path[-1]))
    return final, 100 * final / path_length(gt_path)
