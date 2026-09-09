"""
Does a bounded, well-spread subset of correspondences fit as well as all of them?

2900 matches per edge is far more than a 3-point minimal-set fit needs,
and far more than a Transformer can attend over (4800 cells x 122 frames = 585k
tokens). Farthest-point sampling maximises spatial spread, which is what
conditions the fit -- so the question is how small K can get before accuracy
moves.

`None` means every point, which is the honest top of the sweep: hardcoding a
count would break on any edge whose depth mask kept fewer.

    .venv/bin/python -m experiments.sample_sweep
"""
import numpy as np

from experiments.cpp_testing import GAP, edge, plot
from visual_pose import _geometry as cpp
from visual_pose.checkpoints import load_model
from visual_pose.config import Config
from visual_pose.data_utils.constants import REPO_DIR
from visual_pose.data_utils.dataset import TartanAirSequence
from visual_pose.evaluation.metrics import pose_error
from visual_pose.evaluation.tables import report, worst_edges

SAMPLE_SIZES = [64, 128, 256, 512]
# `rand` is the control that makes the FPS rows mean something. Without it a
# sweep only shows "fewer points is worse", which cannot distinguish losing
# SPREAD from losing COUNT -- and spread is the entire argument for FPS.
METHODS = ["fps", "rand"]
COLOURS = {"fps": ["#4c1d95", "#6d28d9", "#8b5cf6", "#c4b5fd"],
           "rand": ["#7f1d1d", "#b91c1c", "#ef4444", "#fca5a5"]}


def label(method, k):
    return f"{method} {k}"


NAMES = [label(m, k) for m in METHODS for k in SAMPLE_SIZES] + ["all"]
# every row is a metric rigid fit, so t is in metres and chain() must NOT
# rescale by ground-truth |t| -- that would hand back the scale being measured
METRIC = set(NAMES)
PALETTE = {label(m, k): COLOURS[m][i]
           for m in METHODS for i, k in enumerate(SAMPLE_SIZES)}
PALETTE["all"] = "#111827"


def main():
    cpp.set_seed(0)         # FPS seeds from the RNG too; pin it or K rows wander
    cfg = Config()
    cfg.run_name = "k7_d122_mma70"
    seq = TartanAirSequence(REPO_DIR / "data" / "tartan_air" / cfg.val_sequence)
    model = load_model(cfg).eval()

    frames = list(range(0, len(seq), GAP))
    print(f"{cfg.val_sequence}: {len(frames) - 1} edges, gap {GAP}, "
          f"checkpoint {cfg.run_name}")

    poses = {name: [] for name in NAMES}
    gt, kept = [], []
    for n, (i, j) in enumerate(zip(frames[:-1], frames[1:])):
        _, _, p_i, p_j, T_ji = edge(seq, model, i, j)
        gt.append(T_ji)
        kept.append(len(p_i))
        rng = np.random.default_rng(n)      # per-edge seed, so runs replay

        def fit(idx):
            # ONE index set, applied to BOTH clouds. Sampling them separately
            # picks different points and silently destroys the correspondence.
            return cpp.kabsch_ransac(p_i[idx], p_j[idx]).T_ji.matrix()

        poses["all"].append(fit(np.arange(len(p_i))))
        for k in SAMPLE_SIZES:
            take = min(k, len(p_i))
            poses[label("fps", k)].append(fit(cpp.farthest_point_sample(p_i, take)))
            poses[label("rand", k)].append(
                fit(rng.choice(len(p_i), take, replace=False)))

    print(f"matches per edge: min {min(kept)}, median {int(np.median(kept))}, "
          f"max {max(kept)}")

    errors = {name: [pose_error(T, T_gt) for T, T_gt in zip(p, gt)]
              for name, p in poses.items()}
    report(errors, poses, gt, METRIC)
    worst_edges(errors, frames, matches=kept)
    plot(poses, gt, errors,
         REPO_DIR / "experiments" / "sample_sweep.png", PALETTE, METRIC)


if __name__ == "__main__":
    main()
