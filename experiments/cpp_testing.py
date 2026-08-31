"""
Compare pose estimators on one sequence, edge by edge.

Every estimator sees the SAME correspondences and the same depth-filtered
point clouds, so the table compares estimators and nothing else.

To add one: write `(uv_i, uv_j, P_i, P_j) -> 4x4 T_ji or None` and put it in
ESTIMATORS. Everything else is shared.

    .venv/bin/python -m experiments.cpp_testing
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from visual_pose import _geometry as cpp
from visual_pose.checkpoints import load_model
from visual_pose.config import Config
from visual_pose.data_utils.constants import DEVICE, INTRINSICS_TARTAN_AIR, REPO_DIR
from visual_pose.data_utils.dataset import TartanAirSequence
from visual_pose.geometry.best_match import get_matching_pairs
from visual_pose.geometry.true_correspondences import back_project

K = INTRINSICS_TARTAN_AIR
K_INV = np.linalg.inv(K)
GAP = 5
# TartanAir returns ~1e4 m for sky; those points dominate any least-squares fit
MAX_DEPTH = 100.0


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------

def eight_point(uv_i, uv_j, P_i, P_j):
    """
    Our E solver, decomposed by OpenCV.

    recoverPose does the cheirality check that picks one of the four candidate
    (R, t) -- that is VIS-012, still ours to write. t is a DIRECTION only.
    """
    if len(uv_i) < 8:
        return None
    rays_i = (K_INV @ np.c_[uv_i, np.ones(len(uv_i))].T).T
    rays_j = (K_INV @ np.c_[uv_j, np.ones(len(uv_j))].T).T
    E = cpp.eight_point_algorithm(rays_i, rays_j)
    _, R, t, _ = cv2.recoverPose(E, uv_i, uv_j, K)
    return np.block([[R, t.reshape(3, 1)], [np.zeros((1, 3)), 1.0]])


def kabsch(uv_i, uv_j, P_i, P_j):
    """Exact least squares on all points -- no outlier rejection at all."""
    if len(P_i) < 3:
        return None
    return cpp.kabsch_algorithm(P_i, P_j).T_ji


def ransac_3d(uv_i, uv_j, P_i, P_j):
    """Kabsch inside RANSAC: 3-point samples, metres-valued inlier threshold."""
    if len(P_i) < 3:
        return None
    fit = cpp.kabsch_ransac(P_i, P_j)
    return None if not len(fit.inliers) else fit.T_ji


ESTIMATORS = {
    "eight point": eight_point,
    "kabsch": kabsch,
    "kabsch + 3d ransac": ransac_3d,
}
# t in metres, or direction only? Decides the |t| error column and the chaining.
METRIC = {"kabsch", "kabsch + 3d ransac"}
PALETTE = {"eight point": "#dc2626", "kabsch": "#f97316",
           "kabsch + 3d ransac": "#7c3aed"}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def depth_at(depth, uv):
    """
    Bilinear depth lookup. Matches are subpixel and |t| comes from depth alone,
    so rounding costs directly: vs nearest neighbour, |t| error p50 0.74 -> 0.62
    cm, worst edge 3.42 -> 2.00 deg (experiments/depth_sampling.py).
    """
    H, W = depth.shape
    u0 = np.clip(np.floor(uv[:, 0]).astype(int), 0, W - 2)
    v0 = np.clip(np.floor(uv[:, 1]).astype(int), 0, H - 2)
    fu = np.clip(uv[:, 0] - u0, 0, 1)
    fv = np.clip(uv[:, 1] - v0, 0, 1)
    top = depth[v0, u0] * (1 - fu) + depth[v0, u0 + 1] * fu
    bot = depth[v0 + 1, u0] * (1 - fu) + depth[v0 + 1, u0 + 1] * fu
    return top * (1 - fv) + bot * fv


@torch.no_grad()
def edge(seq, model, i, j):
    """One frame pair -> shared correspondences, clouds, and ground truth."""
    def tensor(rgb):
        return torch.from_numpy(rgb).permute(2, 0, 1).float().div(255)[None].to(DEVICE)

    uv_i, uv_j = get_matching_pairs(model, tensor(seq.rgb(i)), tensor(seq.rgb(j)))
    uv_i = uv_i.cpu().numpy().astype(np.float64)
    uv_j = uv_j.cpu().numpy().astype(np.float64)

    d_i, d_j = depth_at(seq.depth(i), uv_i), depth_at(seq.depth(j), uv_j)
    ok = ((d_i > 0) & (d_j > 0) & (d_i < MAX_DEPTH) & (d_j < MAX_DEPTH)
          & np.isfinite(d_i) & np.isfinite(d_j))

    # every estimator gets the same subset, so depth validity is not a variable
    uv_i, uv_j, d_i, d_j = uv_i[ok], uv_j[ok], d_i[ok], d_j[ok]
    T_ji = np.linalg.inv(seq.pose(j)) @ seq.pose(i)      # camera i -> camera j
    return uv_i, uv_j, back_project(uv_i, d_i, K), back_project(uv_j, d_j, K), T_ji


def pose_error(T_est, T_gt):
    """(rotation deg, translation direction deg, |t| error cm)."""
    if T_est is None:
        return np.nan, np.nan, np.nan
    dR = T_est[:3, :3] @ T_gt[:3, :3].T
    rot = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    t, t_gt = T_est[:3, 3], T_gt[:3, 3]
    cos = abs(t @ t_gt) / (np.linalg.norm(t) * np.linalg.norm(t_gt))
    return (rot, np.degrees(np.arccos(np.clip(cos, -1, 1))),
            100 * abs(np.linalg.norm(t) - np.linalg.norm(t_gt)))


def chain(poses, gt, metric):
    """
    Fold per-edge relative poses into a world path. Positions only.

    metric=False borrows |t| from ground truth per edge: E recovers a direction
    and nothing more, so without it the path is meaningless. The rigid rows earn
    their scale from depth and use t as given -- that is the point of them.

    Diagnostic version. VIS-024 is the real one.
    """
    T = np.eye(4)
    out = [T[:3, 3].copy()]
    for T_est, T_gt in zip(poses, gt):
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


def plot(poses, gt, errors, frames, path):
    fig = plt.figure(figsize=(17, 5))
    gt_pos = chain(gt, gt, metric=True)

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.plot(*gt_pos.T, "-", color="#111827", label="ground truth", lw=1.8)
    ax.scatter(*gt_pos[0], color="#16a34a", s=40, label="start")
    for name, p in poses.items():
        est = chain(p, gt, name in METRIC)
        ax.plot(*est.T, "--", color=PALETTE[name], label=name, lw=1.8)
    ax.set_title("camera trajectory\n(non-metric edges rescaled by GT |t|)", fontsize=9)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend(fontsize=7)

    for k, title in enumerate(["rotation error per edge (deg)",
                               "translation direction error per edge (deg)"]):
        a = fig.add_subplot(1, 3, k + 2)
        for name, err in errors.items():
            # clamp for the log axis: exact zeros would vanish entirely
            a.plot(np.maximum(np.array(err)[:, k], 1e-4), ".-",
                   color=PALETTE[name], label=name, ms=4, lw=1)
        a.set_title(title, fontsize=9); a.set_xlabel("edge"); a.set_yscale("log")
        a.grid(alpha=0.3); a.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print(f"\nwrote {path}")


# --------------------------------------------------------------------------

def main():
    cpp.set_seed(0)         # RANSAC is randomized; pin it or the numbers wander
    cfg = Config()
    cfg.run_name = "k7_d122_mma70"
    seq = TartanAirSequence(REPO_DIR / "data" / "tartan_air" / cfg.val_sequence)
    model = load_model(cfg).eval()

    frames = list(range(0, len(seq), GAP))
    print(f"{cfg.val_sequence}: {len(frames) - 1} edges, gap {GAP}, "
          f"checkpoint {cfg.run_name}")

    poses = {name: [] for name in ESTIMATORS}
    gt = []
    for i, j in zip(frames[:-1], frames[1:]):
        *inputs, T_ji = edge(seq, model, i, j)
        gt.append(T_ji)
        for name, estimate in ESTIMATORS.items():
            poses[name].append(estimate(*inputs))

    errors = {name: [pose_error(T, T_gt) for T, T_gt in zip(p, gt)]
              for name, p in poses.items()}

    print(f"\n{'estimator':>20} | {'rotation (deg)':^23} | {'t dir':>6} | {'|t| cm':>7}")
    print(f"{'':>20} | {'p50':>7} {'p90':>7} {'max':>7} | {'p50':>6} | {'p50':>7}")
    for name, err in errors.items():
        e = np.array(err)
        r = np.nanpercentile(e[:, 0], [50, 90])
        mag = f"{np.nanmedian(e[:, 2]):>7.2f}" if name in METRIC else f"{'-':>7}"
        print(f"{name:>20} | {r[0]:>7.3f} {r[1]:>7.3f} {np.nanmax(e[:, 0]):>7.3f} | "
              f"{np.nanmedian(e[:, 1]):>6.2f} | {mag}")

    # rotation compounds down a chain and translation direction does not, so the
    # tail predicts drift far better than any median
    gt_pos = chain(gt, gt, metric=True)
    length = np.linalg.norm(np.diff(gt_pos, axis=0), axis=1).sum()
    print(f"\n{'estimator':>20} | >5 deg | failed | final err | drift  "
          f"(over {length:.1f} m)")
    for name, err in errors.items():
        e = np.array(err)
        final = np.linalg.norm(chain(poses[name], gt, name in METRIC)[-1] - gt_pos[-1])
        print(f"{name:>20} | {int(np.nansum(e[:, 0] > 5)):>6} | "
              f"{int(np.isnan(e[:, 0]).sum()):>6} | {final:>7.2f} m | "
              f"{100 * final / length:>5.1f}%")

    plot(poses, gt, errors, frames,
         REPO_DIR / "experiments" / "eight_point_trajectory.png")


if __name__ == "__main__":
    main()
