"""
Relative pose for one edge: two point clouds -> T_ji, camera i into camera j.

All estimators are `(P_i, P_j, cfg) -> cpp.RansacFit | None`, so
WorldModel.get_relative_pose takes any of them. None means no usable fit.

`inliers` is which points the pose was fitted to. `inlier_ratio` is NaN when no
agreement test was run, so a caller gating on it must treat NaN as "unknown"
rather than as zero.
"""
from __future__ import annotations

import cv2
import numpy as np

from visual_pose import _geometry as cpp
from visual_pose.config import Config


def kabsch(P_i: np.ndarray, P_j: np.ndarray, cfg: Config) -> cpp.RansacFit | None:
    """
    Least squares on every point, no outlier rejection.

    Degenerate on collinear points but fine on coplanar ones, which is why depth
    beats the epipolar route: a flat wall is an ordinary input here.
    """
    if len(P_i) < 3:
        return None
    fit = cpp.kabsch_algorithm(P_i, P_j)
    return cpp.RansacFit(fit.T_ji, range(len(P_i)), np.nan)


def kabsch_ransac(P_i: np.ndarray, P_j: np.ndarray, cfg: Config) -> cpp.RansacFit | None:
    """
    Kabsch inside RANSAC. The C++ side owns the sampling and the residual.

    Returns the whole fit, not just the pose, so the edge weight is computed
    from the same samples the pose was.
    """
    if len(P_i) < 3:
        return None
    fit = cpp.kabsch_ransac(P_i, P_j,
                            inlier_threshold=cfg.inlier_threshold,
                            degeneracy_threshold=cfg.degeneracy_threshold)
    # empty inliers is the C++ "no consensus found"
    return fit if len(fit.inliers) else None


def eight_point(P_i: np.ndarray, P_j: np.ndarray, cfg: Config) -> cpp.RansacFit | None:
    """Essential matrix over all correspondences, decomposed to (R, t). Needs >= 8.

    Not a usable pose graph edge: |t| is unrecoverable from the epipolar
    constraint, so this only holds a row in the estimator comparison.

    Takes RAYS, not pixels -- pixels give the fundamental matrix, which fits the
    constraint equally well and decomposes to nonsense. Rays come free from the
    metric points, since P / P_z = K^-1 [u, v, 1].

    On a planar scene E = [v]x H holds for any v, so a family of poses scores
    identically: 98% inliers for poses 66 degrees apart. Missing information,
    not outliers, so RANSAC cannot help.
    """
    if len(P_i) < 8:
        return None
    ray_i, ray_j = P_i / P_i[:, 2:3], P_j / P_j[:, 2:3]
    E = cpp.eight_point_algorithm(ray_i, ray_j)

    # cv2's no-cameraMatrix overload assumes focal 1 and principal point 0,
    # which is what rays already are. Picks one of E's four (R, t) by cheirality.
    _, R, t, _ = cv2.recoverPose(E, ray_i[:, :2].copy(), ray_j[:, :2].copy())

    # the 4th return is a cheirality mask with a distance cutoff, not an
    # agreement test -- 25 of 150 exact correspondences fail it
    T_ji = np.eye(4)
    T_ji[:3, :3], T_ji[:3, 3] = R, t.ravel()
    return cpp.RansacFit(cpp.PoseSE3(T_ji), range(len(P_i)), np.nan)
