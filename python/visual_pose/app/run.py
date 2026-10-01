"""
Drive WorldModel over one or more scenes and watch it in the GUI.

    .venv/bin/python -m visual_pose.viz.rerun_app  # leave open
    .venv/bin/python -m visual_pose.app.run        # edit the block below, re-run

A variant is (which solvers) plus (a Config), and every parameter lives on the
Config that training uses. `sweep` turns any field of either into a list.

Each stage reports rather than raising, so a stub names itself instead of
killing the run.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field, fields, replace
from enum import StrEnum
from functools import partial
from typing import Callable

import numpy as np
import torch

from visual_pose import _geometry as cpp
from visual_pose.checkpoints import load_model
from visual_pose.config import Config
from visual_pose.data_utils.hm3d_sequence import HM3D, POSES_FILE
from visual_pose.data_utils.sources import source
from visual_pose.evaluation import metrics
from visual_pose.geometry import relative_pose
from visual_pose.models.global_descriptor import avg_pooling, gem_pooling
from visual_pose.pose_graph.world_model import WorldModel
from visual_pose.viz.run_log import DEFAULT_ROOT, RunWriter


class EdgeSolver(StrEnum):
    """Relative pose from one frame pair."""
    EIGHT_POINT = "eight point"
    KABSCH = "kabsch"


class GraphSolver(StrEnum):
    """What runs on the graph of chained edges. NONE = frontend only."""
    NONE = "none"
    GAUSS_NEWTON = "gauss newton"
    # TRANSFORMER = "transformer"     # Phase 3: add one entry to GRAPH_SOLVERS


# Not frozen: Config is not either, so a generated __hash__ would raise.
@dataclass
class Variant:
    """Which solvers to run, and the Config to run them with."""
    edge_solver: EdgeSolver = EdgeSolver.KABSCH
    ransac: bool = True                    # which robust estimator, not a wrapper
    graph_solver: GraphSolver = GraphSolver.NONE
    config: Config = field(default_factory=Config)
    pooling_fn: Callable[[torch.Tensor], torch.Tensor] = gem_pooling

    @property
    def metric(self) -> bool:
        """t in metres? A direction-only t cannot feed a graph solver."""
        return self.edge_solver in METRIC

    @property
    def wants_graph(self) -> bool:
        return self.graph_solver is not GraphSolver.NONE

    @property
    def edge_function(self):
        """(P_i, P_j, cfg) -> RansacFit | None, for (edge_solver, ransac)."""
        try:
            return EDGE_SOLVERS[(self.edge_solver, self.ransac)]
        except KeyError:
            raise SystemExit(
                f"no estimator for edge_solver={self.edge_solver!s}, "
                f"ransac={self.ransac}\n"
                f"  have: {sorted((str(s), r) for s, r in EDGE_SOLVERS)}\n"
                "  see geometry/relative_pose.py -- then one line in EDGE_SOLVERS"
            ) from None

    @property
    def graph_function(self):
        """(graph) -> None, mutating it in place. None when no graph is wanted."""
        if not self.wants_graph:
            return None
        try:
            return GRAPH_SOLVERS[self.graph_solver](self.config)
        except KeyError:
            raise SystemExit(f"no graph solver registered for "
                             f"{self.graph_solver!s}; add it to GRAPH_SOLVERS") from None

    def check(self) -> None:
        """Fail at import, not at edge 40 of a long run."""
        _ = self.edge_function
        if self.wants_graph:
            _ = self.graph_function
        if self.wants_graph and not self.metric:
            raise SystemExit(
                f"{solver_name(self)} recovers a direction only, so a graph "
                "built from it means nothing -- drop graph_solver on that variant")


def sweep(variants, field_name: str, values) -> list[Variant]:
    """
    One variant (or list) times a list of values for one field.

        RUNS = {"P000": sweep(COMMON[0], "loop_min_inliers", [10, 25, 40, 60])}
        sweep(sweep(COMMON[0], "val_frame_gap", [5, 10]), "huber", [0.0, 0.5])

    Works on Variant and Config fields alike, and nests for a grid. The swept
    field lands in every label automatically, because it now varies.
    """
    base = [variants] if isinstance(variants, Variant) else list(variants)
    on_variant = {f.name for f in fields(Variant)} - {"config"}
    on_config = {f.name for f in fields(Config)}
    if field_name not in on_variant | on_config:
        raise SystemExit(f"cannot sweep {field_name!r}\n"
                         f"  Variant: {sorted(on_variant)}\n"
                         f"  Config:  {sorted(on_config)}")
    out = []
    for variant in base:
        for value in values:
            out.append(replace(variant, **{field_name: value})
                       if field_name in on_variant
                       else replace(variant, config=replace(
                           variant.config, **{field_name: value})))
    return out


def hm3d(scene: str = "*", trajectory: str = "trajectory_*") -> list[str]:
    """
    RUNS keys for every HM3D trajectory on disk that has a poses.txt. Globs:

        hm3d()                              # everything
        hm3d("00801-*")                     # one scene
        hm3d("00801-*", "trajectory_2[0-5]")
    """
    keys = [f"hm3d:{path.parent.parent.parent.name}/{path.parent.name}"
            for path in sorted(HM3D.glob(f"{scene}/trajectories/{trajectory}/{POSES_FILE}"))]
    if not keys:
        # Or a typo'd glob quietly runs nothing.
        raise SystemExit(f"no HM3D trajectories match {scene}/{trajectory} under {HM3D}")
    return keys


# ===== EDIT ME ==============================================================

# What everything below starts from. Anything not named here keeps Config's
# default, and labels name whatever differs.
BASE = Config(
    run_name="fine_tuning_wide_baseline",     # which descriptor weights
    max_depth=100.0,              # 30 is the TRAINING filter; evaluation has
                                  # always run at 100 (TartanAir gives ~1e4 for sky)
    max_edges=None,                 # None = the whole sequence
    huber=2.0,
    similarity_threshold=0.8,
    loop_retrieval_similarity=0.94,
    loop_min_inliers=25,
    val_frame_gap=2,
    loop_min_gap=50,
    loop_max_candidates=500,
    # How the candidate budget is spent. P006, otherwise identical settings:
    #   topk    spans collapse to a 16-kf band (153-169)     ATE 0.268
    #   random  spans 54-147, 17/500 verified                ATE 0.266
    #   span    spans 64-169, 153/500 verified               ATE 0.257
    # Does NOT transfer: same config gives -48% on P001, -3.5% on P000 and
    # +2.2% on P002, which has no true revisit beyond span 100.
    loop_selection="span",
    loop_span_bands=8,
)

COMMON = [
    Variant(EdgeSolver.KABSCH, ransac=True,
            graph_solver=GraphSolver.GAUSS_NEWTON, config=BASE, pooling_fn=gem_pooling),
    # Variant(EdgeSolver.KABSCH, ransac=False, config=BASE, graph_solver=GraphSolver.GAUSS_NEWTON),
    # Variant(EdgeSolver.KABSCH, ransac=False, config=BASE),
    # Variant(EdgeSolver.EIGHT_POINT, ransac=False, config=BASE),
]

# scene -> variants to run on that scene
RUNS = {
    # HM3D, same pipeline and metrics as TartanAir. Key: hm3d:<scene dir>/<trajectory dir>
    # 2026-09-30: all 20 trajectories (trajectory_00-09 on 00800 and 00801)
    # regenerated on a navmesh rebuilt for a 0.30 m / 1.80 m agent, so no
    # frame is within 15 cm of geometry. The earlier blank-wall and
    # wall-aliasing cases (old trajectory_20 and _23, 00801) are gone with
    # the old data; the gates built on them stay.
    # Everything on disk. Comment out, and uncomment below, to pick by hand.
    # **dict.fromkeys(hm3d(), COMMON),

    # iPhone LiDAR rosbag (EECE5550): depth only, normal image in place of RGB,
    # no ground truth. Key: bag:<directory under BAGS>. 216 frames at 4 Hz.
    "bag:inputs_20250410_203244": COMMON,
    # **dict.fromkeys(hm3d("00803-*"), COMMON),               # one scene
    # "hm3d:00801-HaxA7YrQdEC/trajectory_00": COMMON,         # named ones
    # "hm3d:00800-TEEsavR23oF/trajectory_01": COMMON,

    # "P006": COMMON,
    # "P005": COMMON,
    # "P004": COMMON,
    # "P003": COMMON,
    # "P002": COMMON,
    # "P001": COMMON,
    # "P000": COMMON,

    # ---- sweeps run 2026-09-09, P006 unless noted ---------------------------
    # selection policy, everything else at BASE:
    #   "P006": sweep(COMMON[0], "loop_selection", ["topk", "random", "span"]),
    #
    # candidate budget. U-curve: too few under-constrains, too many stacks
    # correlated edges on the same revisit event and the graph over-trusts them.
    #   25 -> 0.291   100 -> 0.275   200 -> 0.262   500 -> 0.257
    #   1500 -> 0.285  3000 -> 0.303   (all closures accurate, 0-1 over 5 deg)
    #   "P006": sweep(COMMON[0], "loop_max_candidates", [25, 100, 200, 500, 1500, 3000]),
    #
    # huber: identical ATE at 0, 2, 5. There are no outlier closures for it to
    # suppress; it is the wrong tool for correlated ones.
    #   "P006": sweep(COMMON[0], "huber", [0.0, 2.0, 5.0]),
    #
    # loop_retrieval_similarity: byte-identical results at 0.94 and 0.98. The
    # fingerprint gives 5.2% precision against a 5.2% base rate, so the
    # threshold admits the whole pool and verification does the discriminating.
    #   "P006": sweep(COMMON[0], "loop_retrieval_similarity", [0.94, 0.98]),
    #
    # bands x span floor: lgap50 beat lgap100 everywhere (0.257-0.264 vs
    # 0.264-0.271) -- the 50-100 band holds ~500 genuine revisit pairs.
    #   "P006": sweep(sweep(COMMON[0], "loop_span_bands", [3, 5, 8]),
    #                 "loop_min_gap", [50, 100]),
}

# ---- solver -----------------------------------------------------------------
SOLVER_TOLERANCE = 1e-8       # gauss_newton's own |delta| stop
SOLVER_MAX_STEPS = 100        # animation only: cap on stepped iterations
SOLVER_CONVERGED = 1e-10      # animation only: stop when positions stop moving
SOLVER_VERBOSE = False        # C++ per-iteration cost; prints out of order with
                              # Python, so it lands at the end of the output

# ---- viewer -----------------------------------------------------------------
CLEAR_FIRST = True     # wipe the window before this invocation draws
GRAPH_DELAY = 0.0     # seconds between animated solver iterations; 0 = batch: one C++ solve, no animation
ROOT = DEFAULT_ROOT    # where the logs go; the viewer watches it
# Anything non-solid. The viewer reserves SOLID for the ground-truth reference,
# so an estimate can never be mistaken for truth; the frontend tracks take the
# viewer's dotted default and the graph twin needs to differ from them.
GRAPH_LINESTYLE = "--"

# ===========================================================================

# (edge_solver, ransac) -> (P_i, P_j, cfg) -> RansacFit | None
EDGE_SOLVERS = {
    (EdgeSolver.KABSCH, False): relative_pose.kabsch,
    (EdgeSolver.KABSCH, True): relative_pose.kabsch_ransac,
    (EdgeSolver.EIGHT_POINT, False): relative_pose.eight_point,
    # (EdgeSolver.EIGHT_POINT, True): relative_pose.eight_point_ransac,
}

# Whose t is in metres. Decides the |t| column, whether chaining borrows a
# scale, and whether a graph built from them means anything.
METRIC = {EdgeSolver.KABSCH}

# graph_solver -> (config) -> SolveFn. The C++ method already matches the
# contract, so no wrapper. The transformer goes here.
GRAPH_SOLVERS = {
    GraphSolver.GAUSS_NEWTON: lambda config: partial(
        cpp.FactorGraph.gauss_newton, iter_threshold=SOLVER_TOLERANCE,
        verbose=SOLVER_VERBOSE, huber=config.huber, lm_lambda_init=config.lm_lambda_init),
}

# How each field appears in a label, when it is one of the things that varies.
DESCRIBE = {
    "run_name": lambda x: str(x),
    "max_edges": lambda x: f"n{x}",
    "max_depth": lambda x: f"d{x:g}",
    "similarity_threshold": lambda x: f"sim{x:g}",
    "seed": lambda x: f"seed{x}",
    "loop_min_gap": lambda x: f"lgap{x}",
    "loop_min_inliers": lambda x: f"linl{x}",
    "loop_retrieval_similarity": lambda x: f"lret{x:g}",
    "loop_max_candidates": lambda x: f"lcap{x}",
    "loop_selection": lambda x: f"sel-{x}",
    "loop_span_bands": lambda x: f"bands{x}",
    "weight_edges": lambda x: "weighted" if x else "unweighted",
    "huber": lambda x: f"huber{x:g}",
}
FRONTEND_FIELDS = ("run_name", "max_edges", "max_depth", "similarity_threshold",
                   "seed", "loop_min_gap", "loop_min_inliers", "loop_retrieval_similarity",
                   "loop_max_candidates", "loop_selection", "loop_span_bands")
GRAPH_FIELDS = ("weight_edges", "huber")

SCENES = {scene: list(variants) for scene, variants in RUNS.items()}
EVERY = [v for variants in SCENES.values() for v in variants]
if not EVERY:
    raise SystemExit("RUNS is empty")

# Name only what varies, or labels are unreadable. BASELINE is Config's own
# defaults, not BASE: otherwise a label only separates variants within one
# process, and re-running after an edit gives two identically-named lines.
BASELINE = Config()
VARIES = {name: len({getattr(v.config, name) for v in EVERY}) > 1
          for name in DESCRIBE}


def solver_name(variant: Variant) -> str:
    return str(variant.edge_solver) + (" + ransac" if variant.ransac else "")


def described(config: Config, names) -> list[str]:
    """Name a field if it varies within this run, or deviates from the default."""
    out = []
    for name in names:
        value = getattr(config, name)
        if value is None:
            continue
        if VARIES[name] or value != getattr(BASELINE, name):
            out.append(DESCRIBE[name](value))
    return out


def frontend_label(variant: Variant, scene: str) -> str:
    """
    Names the estimate, not the variant, so one frontend is one line however
    many graph solvers run on top of it.
    """
    parts = [solver_name(variant), scene, f"g{variant.config.val_frame_gap}"]
    parts += described(variant.config, FRONTEND_FIELDS)
    if variant.config.note:
        parts.append(variant.config.note)
    return " - ".join(parts)


def twin_label(variant: Variant, front: str) -> str:
    """The optimised line hanging off `front`."""
    parts = [str(variant.graph_solver)] + described(variant.config, GRAPH_FIELDS)
    return f"{front} + {' '.join(parts)}"


for _v in EVERY:
    _v.check()


def ground_truth(sequence, frames: list[int]) -> list[np.ndarray]:
    """T_ji per consecutive keyframe pair: camera i into camera j."""
    return edge_truth(sequence, list(zip(frames, frames[1:])))


def edge_truth(sequence, pairs: list[tuple[int, int]]) -> list[np.ndarray]:
    """T_ji for arbitrary (i, j) pairs, so bridged edges score against their own
    truth rather than against the consecutive pair they replaced."""
    return [np.linalg.inv(sequence.pose(j)) @ sequence.pose(i) for i, j in pairs]


def keyframes(sequence, gap: int, max_edges: int | None) -> list[int]:
    stop = len(sequence) if max_edges is None else min(len(sequence),
                                                       (max_edges + 1) * gap)
    return list(range(0, stop, gap))


def animate_solve(world, run, track: str, *, huber: float, lm_lambda_init: float) -> None:
    """
    Step the solver one iteration at a time so the viewer sees it converge.

    iter_threshold=1e30 exits the C++ loop after one iteration. Stepping lives
    here, not in SolveFn: only the animation wants the intermediate states.
    Pass lm_lambda_init=0: Levenberg-Marquardt needs its lambda and previous
    cost to survive between iterations, which one call per iteration cannot do.
    """
    graph = world.factor_graph
    world.calibrate()                       # optimize() would, but we bypass it
    previous = None
    try:
        for k in range(SOLVER_MAX_STEPS):
            cpp.FactorGraph.gauss_newton(graph, iter_threshold=1e30,
                                         verbose=SOLVER_VERBOSE, huber=huber, lm_lambda_init=lm_lambda_init)
            positions = np.array(graph.poses())[:, :3, 3]
            run.frame(positions, track=track, x=k + 1, x_label="iteration")
            if GRAPH_DELAY:
                time.sleep(GRAPH_DELAY)
            if previous is not None and np.abs(positions - previous).max() < SOLVER_CONVERGED:
                return
            previous = positions
    finally:
        world.rebuild_chain_covariance()    # poses moved; the closure gate's transports go with them


def stage(name: str, call):
    """Run one stage, printing the stub that gates it rather than raising."""
    try:
        return call(), None
    except NotImplementedError as e:
        print(f"    {name:22s} BLOCKED  {e}")
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        where = tb.filename.split("visual_pose/")[-1]
        print(f"    {name:22s} ERROR    {type(e).__name__}: {e}  [{where}:{tb.lineno}]")
    return None, "failed"


@torch.no_grad()
def run_variant(variant: Variant, scene: str, sequence, model, run) -> dict[str, float]:
    """One variant, writing tracks into the scene's shared run log. Returns
    track label -> ATE, for the summary."""
    cfg = variant.config
    # gap 2 on 0.6 m / 13.5 deg HM3D frames put keyframes 1.2 m and 27 deg
    # apart and bridged up to 117 of them (2026-09-30); such data is already at
    # keyframe spacing, so the gap applies to video-rate sequences only
    frames = keyframes(sequence, 1 if sequence.frames_are_keyframes else cfg.val_frame_gap, cfg.max_edges)
    nominal = ground_truth(sequence, frames)
    gt_path = metrics.chain(nominal, nominal, metric=True)

    front = frontend_label(variant, scene)
    print(f"  {front}")

    # the C++ generator is shared, so without this the variants consume each
    # other's draws and reordering RUNS changes every number
    cpp.set_seed(cfg.seed)

    world = WorldModel(model, variant.pooling_fn, sequence, cfg,
                       variant.edge_function, variant.graph_function or (lambda g: None))
    world.encode_keyframes(frames)      # batched; lazy per-frame encoding costs 2x
    run.declare_track(front, show_edges=True)

    # One edge at a time so the viewer grows with it. Bridge rather than
    # refuse: a refused odometry edge leaves its far frame without a vertex and
    # the trajectory forks into two gauge anchors.
    estimates: list[np.ndarray] = []
    pairs: list[tuple[int, int]] = []
    # Consecutive pairs are known up front, so match them batched. A bridged
    # keyframe makes the next pair (anchor, j + 2), which is matched singly.
    world.prefetch_matches(list(zip(frames[:-1], frames[1:])))
    # TODO: Reanchor after consecutive failures
    anchor = frames[0]
    for j in frames[1:]:
        if not world.add_edge(anchor, j):
            continue
        # edges[-1]: add_edge appends, so the new edge is the last one
        estimates.append(
            world.factor_graph.edges[-1].measured_pose.inverse().matrix())
        pairs.append((anchor, j))
        anchor = j
        run.frame(metrics.chain(estimates, edge_truth(sequence, pairs),
                                metric=variant.metric),
                  track=front, x=len(estimates), x_label="edge")
    bridged = len(frames) - 1 - len(estimates)
    print(f"    {'odometry edges':22s} {len(estimates)}/{len(frames) - 1} added"
          + (f", {bridged} keyframe(s) bridged" if bridged else ""))
    gt = edge_truth(sequence, pairs)

    candidates, _ = stage("loop candidates", lambda: world.loop_candidates(frames))
    if candidates is not None:
        world.prefetch_matches(candidates)
        closures: list[tuple[int, int]] = []
        closure_est: list[np.ndarray] = []
        for i, j in candidates:
            if not world.add_edge(i, j, closure=True):
                continue
            closure_est.append(
                world.factor_graph.edges[-1].measured_pose.inverse().matrix())
            closures.append((i, j))
        print(f"    {'loop closures':22s} {len(closures)}/{len(candidates)} verified")
        run.note(f"{len(closures)}/{len(candidates)} loop closures verified")

        if closures and sequence.has_ground_truth:
            # p95 as well as p50: a false closure at wide baseline still scores a
            # high inlier count, so it survives verification and hides in the tail
            # rather than moving the median.
            ce = np.array([metrics.pose_error(T, T_gt) for T, T_gt
                           in zip(closure_est, edge_truth(sequence, closures))])
            for tag, q in (("p50", 50), ("p95", 95)):
                rot, direction, magnitude = np.nanpercentile(ce, q, axis=0)
                print(f"    {'closure err ' + tag:22s} rot {rot:.2f} deg, "
                      f"dir {direction:.2f} deg, |t| {magnitude:.2f} cm")
            run.note(f"closure rot err p50 {np.nanmedian(ce[:, 0]):.2f} deg, "
                     f"p95 {np.nanpercentile(ce[:, 0], 95):.2f} deg")

            # Span in KEYFRAMES, the unit loop_min_gap is expressed in. A
            # closure only corrects drift if it spans enough of the chain for
            # error to have accumulated between its endpoints; spans piled up
            # at the minimum gap add cycles that were never in tension.
            at = {f: k for k, f in enumerate(frames)}
            spans = np.array([abs(at[j] - at[i]) for i, j in closures])
            print(f"    {'closure span (kf)':22s} min {spans.min()}, "
                  f"p50 {int(np.median(spans))}, max {spans.max()}"
                  f"  (chain is {len(frames) - 1} kf, gap {cfg.loop_min_gap})")
            run.note(f"closure span kf min {spans.min()} "
                     f"p50 {int(np.median(spans))} max {spans.max()}")

    chained = metrics.chain(estimates, gt, metric=variant.metric)
    ates: dict[str, float] = {}
    if not sequence.has_ground_truth:
        print(f"    {'odometry path':22s} {metrics.path_length(chained):.2f} m  [no ground truth]")
    else:
        errors = np.array([metrics.pose_error(T, T_gt) for T, T_gt in zip(estimates, gt)])
        with np.errstate(invalid="ignore"):
            print(f"    {'edge error p50':22s} rot {np.nanmedian(errors[:, 0]):.2f} deg, "
                  f"dir {np.nanmedian(errors[:, 1]):.2f} deg, "
                  f"|t| {np.nanmedian(errors[:, 2]):.2f} cm")
        print(f"    {'odometry path':22s} {metrics.path_length(chained):.2f} m "
              f"(ground truth {metrics.path_length(gt_path):.2f} m)"
              + ("" if variant.metric else "  [|t| borrowed from GT]"))
        # chain(gt, gt) rather than gt_path: both fold the SAME pair list, so a
        # bridged edge cannot slide the two paths out of correspondence.
        gt_chain = metrics.chain(gt, gt, metric=True)
        odom_ate = metrics.ate(chained, gt_chain)
        odom_aligned = metrics.ate_aligned(chained, gt_chain)
        # both: anchored is what every earlier number in the notes is; aligned
        # drops the first edge's rotation times the lever arm, which is most of
        # the anchored figure on long paths (see metrics.ate)
        print(f"    {'odometry ATE':22s} {odom_ate:.3f} m anchored, {odom_aligned:.3f} m aligned")
        ates[front] = odom_ate
        run.summary(front, f"ATE {odom_ate:.3f} m ({odom_aligned:.3f} aligned)")
    # chain_poses[k] belongs to the far frame of pair k-1; bridged keyframes have none
    run.poses(front, [frames[0]] + [j for _, j in pairs],
              metrics.chain_poses(estimates, gt, metric=variant.metric))

    if variant.wants_graph:
        track = twin_label(variant, front)
        run.declare_track(track, derived_from=front, linestyle=GRAPH_LINESTYLE)
        # Animate only when someone is watching. Levenberg-Marquardt keeps
        # lambda and the previous cost across iterations; stepping one
        # iteration per C++ call (animate_solve) resets that every step, so
        # the batch path solves in one call, which is also 6 s faster per run.
        if GRAPH_DELAY:
            # undamped on purpose: one iteration per call cannot carry LM's
            # state, and a rejected step reads as "converged" to the animation
            # (P006: 0.348 m damped-stepped vs 0.241 m either proper way)
            solve = lambda: animate_solve(world, run, track, huber=cfg.huber, lm_lambda_init=0.0)
        else:
            def solve():
                world.optimize()        # calibrate, solve to SOLVER_TOLERANCE, rebuild the closure gate's chain
                run.frame(np.array(world.factor_graph.poses())[:, :3, 3], track=track, x=1, x_label="iteration")
        _, failed = stage("optimize", solve)
        if not failed:
            positions = np.array(world.factor_graph.poses())[:, :3, 3]
            print(f"    {'optimize':22s} {world.factor_graph}, "
                  f"path {metrics.path_length(positions):.2f} m")
            # Vertices are created in chain order, but read the mapping rather
            # than assuming vertex k is frames[k]: bridged keyframes have none.
            order = sorted(world.vertex_index, key=world.vertex_index.get)
            if sequence.has_ground_truth:
                into_v0 = np.linalg.inv(sequence.pose(order[0]))
                gt_vertices = np.array(
                    [(into_v0 @ sequence.pose(f))[:3, 3] for f in order])
                graph_ate = metrics.ate(positions, gt_vertices)
                graph_aligned = metrics.ate_aligned(positions, gt_vertices)
                print(f"    {'ATE odometry -> graph':22s} "
                      f"{odom_ate:.3f} m -> {graph_ate:.3f} m"
                      f"  ({100 * (graph_ate - odom_ate) / odom_ate:+.1f}%)"
                      f"   aligned {odom_aligned:.3f} m -> {graph_aligned:.3f} m")
                run.note(f"ATE {odom_ate:.3f} m -> {graph_ate:.3f} m (aligned {odom_aligned:.3f} -> {graph_aligned:.3f})")
                ates[track] = graph_ate
                run.summary(track, f"ATE {graph_ate:.3f} m ({graph_aligned:.3f} aligned)")
            run.poses(track, order, np.array(world.factor_graph.poses()))

    run.end(front)
    return ates


def ate_table(ates: dict[tuple[str, str], float]) -> str:
    """(scene, track kind) -> ATE, as rows of scenes and columns of track kinds,
    with mean / std / best / worst per column. Printed and sent to the viewer."""
    scenes = list(dict.fromkeys(scene for scene, _ in ates))
    kinds = list(dict.fromkeys(kind for _, kind in ates))
    label_w = max(len(f"#{len(scenes)} {s}") for s in scenes)

    def row(label, cells, tags=None):
        # number, then a 5-wide slot for best/worst's "#row", so numbers align
        tags = tags or [""] * len(cells)
        return f"{label:<{label_w}}" + "".join(
            f"{c:>9}{t:<5}" for c, t in zip(cells, tags))

    lines = [f"[{k + 1}] {kind}" for k, kind in enumerate(kinds)] + [""]
    lines.append(row("", [f"[{k + 1}]" for k in range(len(kinds))]))
    for r, scene in enumerate(scenes, 1):
        lines.append(row(f"#{r} {scene}", [
            f"{ates[scene, kind]:.3f}" if (scene, kind) in ates else "-"
            for kind in kinds]))
    lines.append("-" * (label_w + 14 * len(kinds)))

    # Per column, over the scenes that produced a number (a failed optimise
    # leaves a hole, not a zero).
    columns = [[(r, ates[scene, kind]) for r, scene in enumerate(scenes, 1)
                if (scene, kind) in ates] for kind in kinds]
    lines.append(row("mean", [f"{np.mean([v for _, v in c]):.3f}" for c in columns]))
    # Sample std (ddof=1): the scenes are a sample of the scenes we could run.
    lines.append(row("std", [f"{np.std([v for _, v in c], ddof=1):.3f}"
                             if len(c) > 1 else "-" for c in columns]))
    for name, pick in (("best", min), ("worst", max)):
        chosen = [pick(c, key=lambda rv: rv[1]) for c in columns]
        lines.append(row(name, [f"{v:.3f}" for _, v in chosen],
                         [f" #{r}" for r, _ in chosen]))
    return "\n".join(lines)


def main() -> None:
    first = CLEAR_FIRST
    ates: dict[tuple[str, str], float] = {}
    for n, (scene, variants) in enumerate(SCENES.items(), 1):
        sequence = source(scene).sequence()
        models: dict[str, object] = {}      # one load per checkpoint, not per variant
        print(f"\n{scene}: {len(sequence)} frames")

        # one log per scene, so the ground truth is written once; a log per
        # variant would draw N overlapping references
        with RunWriter(scene, root=ROOT, meta={
                "scene": scene,
                "variants": [frontend_label(v, scene) for v in variants]}) as run:
            if first:
                run.clear_viewer()
                first = False
            base = variants[0].config
            gt = ground_truth(sequence, keyframes(sequence, base.val_frame_gap,
                                                  base.max_edges))
            # the viewer keys references by label, so each scene needs its own
            if sequence.has_ground_truth:
                run.reference(metrics.chain(gt, gt, metric=True),
                              label=f"ground truth - {scene}")

            for variant in variants:
                name = variant.config.run_name
                if name not in models:
                    loaded = load_model(replace(variant.config, run_name=name)).eval()
                    loaded.device = next(loaded.parameters()).device
                    models[name] = loaded
                for track, ate in run_variant(variant, scene, sequence,
                                              models[name], run).items():
                    # The column is the track minus its scene, so one estimator
                    # lines up across scenes.
                    ates[scene, track.replace(f" - {scene}", "")] = ate

            # Written into the last scene's log, not a log of its own: a
            # separate file replays concurrently and could land before the
            # clear at the top of the first scene's log.
            if n == len(SCENES) and ates:
                table = ate_table(ates)
                print(f"\nATE (m)\n{table}")
                run.table("ATE (m)", table)
        print(f"  {'run log':24s} {run.path.name}")


if __name__ == "__main__":
    main()
