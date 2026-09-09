"""
Compare pose estimators on one sequence, edge by edge.

Every estimator sees the SAME correspondences and the same depth-filtered
point clouds, so the table compares estimators and nothing else.

To add one: write `(uv_i, uv_j, P_i, P_j, fit) -> 4x4 T_ji or None` and put it
in ESTIMATORS. `fit` is one shared kabsch_ransac result -- shared so that the
pose and its information weight come from the SAME random samples, and so the
library's RNG is consumed once per edge rather than once per estimator.

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
from visual_pose.evaluation.metrics import chain, path_length, pose_error
from visual_pose.evaluation.tables import report, report_path, worst_edges
from visual_pose.geometry.best_match import get_matching_pairs
from visual_pose.geometry.true_correspondences import back_project

K = INTRINSICS_TARTAN_AIR
K_INV = np.linalg.inv(K)
GAP = 5
# TartanAir returns ~1e4 m for sky; those points dominate any least-squares fit
MAX_DEPTH = 100.0

# Loop closure candidates. Proposal uses ground-truth proximity as a STAND-IN for
# descriptor retrieval -- but the measurement on an accepted pair comes
# from the frontend, with no ground truth in it. Swap the proposal for descriptor
# nearest-neighbour and nothing downstream changes.
LOOP_MAX_DIST = 2.0     # metres between keyframe positions
LOOP_MIN_GAP = 10       # keyframes apart, so consecutive frames are not "loops"
# Geometric verification is on the ABSOLUTE inlier count, deliberately NOT on the
# inlier ratio. The ratio is the wrong statistic for a loop closure and fails in
# both directions:
#   too low  -- wide-baseline pairs are mostly false matches by nature. Every one
#               of P001's 7 closures scored ratio 0.04-0.12 while being accurate
#               to 0.15-0.81 deg and 6-11 cm. A 0.5 threshold rejected all 7 and
#               left the graph with nothing to optimise.
#   too high -- 3 points are Kabsch's minimal set, so a 3-of-5 fit reports ratio
#               0.60 while being 179 deg wrong. On P002 that let 14 false
#               closures through and made Gauss-Newton oscillate.
# The count separates them cleanly on both sequences: wrong closures had 3-8
# inliers, right ones had 28-163.
LOOP_MIN_INLIER_COUNT = 25
# Loop closures want PRECISION, odometry wants a well-conditioned fit, so they
# do not want the same match filter. Measured on P001 closures: raising the
# cosine threshold 0.70 -> 0.80 halves the measurement error (0.62 -> 0.28 deg,
# 8.3 -> 4.5 cm) at 3.4x fewer matches. Past 0.85 it collapses -- too few
# matches for RANSAC to find a consensus at all. Coupled to the count floor
# above: a stricter threshold lowers every count, so both move together.
LOOP_SIMILARITY = 0.80

# Information weighting. Identity says every edge is equally trustworthy, which
# the data flatly contradicts: on P002 an edge with inlier ratio 0.04 was 84 cm
# wrong while one at 0.84 was 0.33 cm wrong. Set False to recover the old
# behaviour for comparison.
WEIGHT_EDGES = True

cfg = Config()


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------

def eight_point(uv_i, uv_j, P_i, P_j, fit):
    """
    Our E solver, decomposed by OpenCV.

    recoverPose does the cheirality check that picks one of the four candidate
    """
    if len(uv_i) < 8:
        return None
    rays_i = (K_INV @ np.c_[uv_i, np.ones(len(uv_i))].T).T
    rays_j = (K_INV @ np.c_[uv_j, np.ones(len(uv_j))].T).T
    E = cpp.eight_point_algorithm(rays_i, rays_j)
    _, R, t, _ = cv2.recoverPose(E, uv_i, uv_j, K)
    return np.block([[R, t.reshape(3, 1)], [np.zeros((1, 3)), 1.0]])


def kabsch(uv_i, uv_j, P_i, P_j, fit):
    """Exact least squares on all points -- no outlier rejection at all."""
    if len(P_i) < 3:
        return None
    return cpp.kabsch_algorithm(P_i, P_j).T_ji.matrix()


def ransac_3d(uv_i, uv_j, P_i, P_j, fit):
    """Kabsch inside RANSAC: 3-point samples, metres-valued inlier threshold."""
    return None if fit is None or not len(fit.inliers) else fit.T_ji.matrix()


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
def edge(seq, model, i, j, similarity):
    """One frame pair -> shared correspondences, clouds, and ground truth.

    `similarity` is passed rather than read from cfg because closures run at a
    stricter threshold than odometry (LOOP_SIMILARITY).
    """
    def tensor(rgb):
        return torch.from_numpy(rgb).permute(2, 0, 1).float().div(255)[None].to(DEVICE)

    uv_i, uv_j = get_matching_pairs(model, tensor(seq.rgb(i)), tensor(seq.rgb(j)), similarity)
    uv_i = uv_i.cpu().numpy().astype(np.float64)
    uv_j = uv_j.cpu().numpy().astype(np.float64)

    d_i, d_j = depth_at(seq.depth(i), uv_i), depth_at(seq.depth(j), uv_j)
    ok = ((d_i > 0) & (d_j > 0) & (d_i < MAX_DEPTH) & (d_j < MAX_DEPTH)
          & np.isfinite(d_i) & np.isfinite(d_j))

    # every estimator gets the same subset, so depth validity is not a variable
    uv_i, uv_j, d_i, d_j = uv_i[ok], uv_j[ok], d_i[ok], d_j[ok]
    T_ji = np.linalg.inv(seq.pose(j)) @ seq.pose(i)      # camera i -> camera j
    return uv_i, uv_j, back_project(uv_i, d_i, K), back_project(uv_j, d_j, K), T_ji


def plot(poses, gt, errors, path, palette, metric, extra_paths=None):
    """`extra_paths` are already-chained (N, 3) position arrays -- the pose graph
    outputs ABSOLUTE poses, so it has no per-edge list to chain."""
    fig = plt.figure(figsize=(17, 5))
    gt_pos = chain(gt, gt, metric=True)

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.plot(*gt_pos.T, "-", color="#111827", label="ground truth", lw=1.8)
    ax.scatter(*gt_pos[0], color="#16a34a", s=40, label="start")
    for name, p in poses.items():
        est = chain(p, gt, name in metric)
        ax.plot(*est.T, "--", color=palette[name], label=name, lw=1.8)
    for name, (positions, colour) in (extra_paths or {}).items():
        ax.plot(*positions.T, "-", color=colour, label=name, lw=2.2)
    ax.set_title("camera trajectory\n(non-metric edges rescaled by GT |t|)", fontsize=9)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend(fontsize=7)

    for k, title in enumerate(["rotation error per edge (deg)",
                               "translation direction error per edge (deg)"]):
        a = fig.add_subplot(1, 3, k + 2)
        for name, err in errors.items():
            # clamp for the log axis: exact zeros would vanish entirely
            a.plot(np.maximum(np.array(err)[:, k], 1e-4), ".-",
                   color=palette[name], label=name, ms=4, lw=1)
        a.set_title(title, fontsize=9); a.set_xlabel("edge"); a.set_yscale("log")
        a.grid(alpha=0.3); a.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print(f"\nwrote {path}")


def information_scale(P_i, P_j, fit):
    """
    A scalar inverse-variance for one rigid fit, from its OWN residuals.

    Kabsch is least squares, so its pose covariance falls as sigma^2 / N with N
    the inlier count and sigma the per-axis residual spread. Information is the
    inverse, hence N / sigma^2 -- an edge earns trust by having many points that
    agree tightly, and both halves matter: 3 points agreeing perfectly is the
    minimal set fitting itself, not evidence.

    Isotropic, which is a known simplification: rotation is in radians and
    translation in metres, so one scalar cannot express that depth error grows
    with range. Good enough to test whether
    weighting helps at all.
    """
    idx = np.asarray(fit.inliers)
    if len(idx) < 3:
        return 0.0
    R, t = fit.T_ji.rotation().matrix(), fit.T_ji.translation()
    residual = P_j[idx] - (P_i[idx] @ R.T + t)
    variance = max(np.mean(np.sum(residual ** 2, axis=1)) / 3.0, 1e-9)
    return len(idx) / variance


def loop_candidates(seq, frames):
    """Keyframe index pairs that revisit the same place. GT-proposed, see the
    constants above."""
    positions = np.array([seq.pose(f)[:3, 3] for f in frames])
    separation = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    index = np.arange(len(frames))
    keep = ((separation < LOOP_MAX_DIST)
            & (np.abs(index[:, None] - index[None, :]) > LOOP_MIN_GAP))
    a, b = np.where(np.triu(keep))
    return list(zip(a.tolist(), b.tolist()))


@torch.no_grad()
def loop_measurements(seq, model, frames, candidates):
    """Run the real frontend on each candidate; keep the ones it can verify."""
    accepted = []
    for a, b in candidates:
        _, _, P_i, P_j, _ = edge(seq, model, frames[a], frames[b], LOOP_SIMILARITY)
        if len(P_i) < 3:
            continue
        fit = cpp.kabsch_ransac(P_i, P_j)
        if len(fit.inliers) >= LOOP_MIN_INLIER_COUNT:
            accepted.append((a, b, fit.T_ji.matrix(), information_scale(P_i, P_j, fit)))
    return accepted


def build_graph(edge_poses, loops, edge_scales=None):
    """
    Chain the frontend edges into a graph and add the loop closures. No solve.

    Split out of optimize() so a caller can hold the graph and step the solver
    one iteration at a time -- which is how the optimisation gets animated.

    Convention: the solver's residual is Log(Z^-1 T_from^-1 T_to) over ABSOLUTE
    vertex poses, so an edge measurement is Z_ij = T_Wi^-1 T_Wj. The frontend
    returns T_ji (frame i into frame j) = T_Wj^-1 T_Wi, so Z is its INVERSE.
    Getting this backwards is the classic pose-graph bug and it fails silently.
    """
    measurements = [np.linalg.inv(T) for T in edge_poses]

    absolute = [np.eye(4)]
    for Z in measurements:
        absolute.append(absolute[-1] @ Z)

    edge_from = list(range(len(measurements)))
    edge_to = list(range(1, len(measurements) + 1))
    scales = list(edge_scales) if edge_scales is not None else [1.0] * len(measurements)
    for a, b, T_ji, scale in loops:
        edge_from.append(a)
        edge_to.append(b)
        measurements.append(np.linalg.inv(T_ji))
        scales.append(scale)

    if WEIGHT_EDGES and edge_scales is not None:
        # Relative trust between edges, median-centred.
        reference = np.median([s for s in scales if s > 0]) or 1.0
        information = [np.eye(6) * (s / reference) for s in scales]
        information = calibrate(information, absolute, edge_from, edge_to, measurements)
    else:
        information = [np.eye(6)] * len(measurements)
    timestamps = [0.0] * len(absolute)

    graph = cpp.FactorGraph(absolute, timestamps, edge_from, edge_to,
                            measurements, information)
    print(f"\npose graph: {len(graph)} vertices, {len(graph.edges)} edges "
          f"({len(loops)} loop closures)")
    return graph


def se3_log(T):
    """Tangent vector of a 4x4, twist ordering [rho, phi] per the C++ convention."""
    return np.asarray(cpp.PoseSE3(T).log()).ravel()


def calibrate(information, absolute, edge_from, edge_to, measurements):
    """
    Rescale every information matrix by one constant so a typical residual is 1.

    This does NOT change the solution -- H and b both scale, so delta is
    unchanged. What it changes is the MEANING of the Huber threshold. Left
    uncalibrated, `information_scale` only fixes the relative trust between
    edges (median edge -> identity), so the absolute residual scale is whatever
    the frontend's units happen to give: measured 0.018-1.3 on P000, which made
    the textbook delta of 1-3 completely inert and forced a per-dataset sweep.

    Calibrated, delta is in units of "typical closure disagreement" and should
    transfer. Only edges with a non-zero residual enter the median: at the
    initial estimate the odometry chain reproduces its own measurements exactly,
    so those residuals are 0 by construction and would drag the median to 0.
    """
    residuals = []
    for k, (i, j) in enumerate(zip(edge_from, edge_to)):
        error = np.linalg.inv(measurements[k]) @ np.linalg.inv(absolute[i]) @ absolute[j]
        r = se3_log(error)
        residuals.append(np.sqrt(r @ information[k] @ r))

    nonzero = [r for r in residuals if r > 1e-9]
    if not nonzero:
        return information
    # Mahalanobis scales as sqrt(Omega), so to divide the residual by m the
    # information is divided by m^2
    median = float(np.median(nonzero))
    return [W / (median ** 2) for W in information]


def optimize(edge_poses, loops, edge_scales=None):
    """Build, solve to convergence, return (N, 3) world positions."""
    graph = build_graph(edge_poses, loops, edge_scales)
    graph.gauss_newton(1e-8, True)
    return np.array([T[:3, 3] for T in graph.poses()])


# --------------------------------------------------------------------------

def main():
    cpp.set_seed(0)         # RANSAC is randomized; pin it or the numbers wander
    cfg.run_name = "fine_tuning_wide_baseline"
    seq = TartanAirSequence(REPO_DIR / "data" / "tartan_air" / "P002")
    model = load_model(cfg).eval()

    frames = list(range(0, len(seq), GAP))
    print(f"{cfg.val_sequence}: {len(frames) - 1} edges, gap {GAP}, "
          f"checkpoint {cfg.run_name}")

    poses = {name: [] for name in ESTIMATORS}
    gt, edge_scales = [], []
    for i, j in zip(frames[:-1], frames[1:]):
        *inputs, T_ji = edge(seq, model, i, j, cfg.similarity_threshold)
        gt.append(T_ji)
        # ONE ransac per edge, shared by the estimator and the weighting
        fit = cpp.kabsch_ransac(inputs[2], inputs[3]) if len(inputs[2]) >= 3 else None
        for name, estimate in ESTIMATORS.items():
            poses[name].append(estimate(*inputs, fit))
        edge_scales.append(information_scale(inputs[2], inputs[3], fit) if fit else 0.0)

    errors = {name: [pose_error(T, T_gt) for T, T_gt in zip(p, gt)]
              for name, p in poses.items()}

    report(errors, poses, gt, METRIC)
    worst_edges(errors, frames)

    # --- pose graph optimization ----------------------------
    candidates = loop_candidates(seq, frames)
    loops = loop_measurements(seq, model, frames, candidates)
    print(f"\nloop closures: {len(candidates)} proposed, {len(loops)} verified "
          f"(a revisit can face a different direction, so most should fail)")

    gt_pos = chain(gt, gt, metric=True)
    length = path_length(gt_pos)

    # Run it on every metric row: the pose graph removes DRIFT, so the effect is
    # only visible where the frontend left some. The good row is nearly optimal
    # already, which is itself the point.
    extra, shades = {}, {"kabsch": "#059669", "kabsch + 3d ransac": "#0891b2"}
    for name in ("kabsch", "kabsch + 3d ransac"):
        optimized = optimize(poses[name], loops, edge_scales)
        report_path(name, chain(poses[name], gt, True), gt_pos, length)
        report_path(f"{name} + pose graph", optimized, gt_pos, length)
        extra[f"{name} + graph"] = (optimized, shades[name])

    plot(poses, gt, errors,
         REPO_DIR / "experiments" / "eight_point_trajectory.png", PALETTE, METRIC,
         extra_paths=extra)


if __name__ == "__main__":
    main()
