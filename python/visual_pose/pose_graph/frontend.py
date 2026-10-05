"""
The frontend as one call: keyframes -> odometry chain -> verified closures,
leaving an UNSOLVED FactorGraph in a WorldModel. run.py drives it with a
viewer attached and solves afterwards; the HM3D generator drives it headless
and writes the graph to disk (save_graph) as a Phase 3 training sample.
Nothing here knows about graph solvers.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from visual_pose import _geometry as cpp
from visual_pose.config import Config
from visual_pose.data_utils.constants import code_version
from visual_pose.geometry import relative_pose
from visual_pose.models.global_descriptor import gem_pooling
from visual_pose.pose_graph.world_model import WorldModel

GRAPH_FILE = "graph.g2o"      # the FactorGraph
GRAPH_SIDECAR = "graph.npz"   # beside it: what g2o cannot hold (see save_graph)


class EdgeSolver(StrEnum):
    """Relative pose from one frame pair."""
    EIGHT_POINT = "eight point"
    KABSCH = "kabsch"


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


# Not frozen: Config is not either, so a generated __hash__ would raise.
@dataclass
class Variant:
    """Which edge solver to run, and the Config to run it with."""
    edge_solver: EdgeSolver = EdgeSolver.KABSCH
    ransac: bool = True                    # which robust estimator, not a wrapper
    config: Config = field(default_factory=Config)
    pooling_fn: Callable[[torch.Tensor], torch.Tensor] = gem_pooling

    @property
    def metric(self) -> bool:
        """t in metres? A direction-only t cannot feed a graph solver."""
        return self.edge_solver in METRIC

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

    def check(self) -> None:
        """Fail at import, not at edge 40 of a long run."""
        _ = self.edge_function


def solver_name(variant: Variant) -> str:
    return str(variant.edge_solver) + (" + ransac" if variant.ransac else "")


def keyframes(sequence, gap: int, max_edges: int | None) -> list[int]:
    stop = len(sequence) if max_edges is None else min(len(sequence),
                                                       (max_edges + 1) * gap)
    return list(range(0, stop, gap))


@dataclass
class Frontend:
    """
    What build() produces, and grows while it runs. The graph in `world` is
    unsolved; its edges are the odometry pairs in order, then the closures,
    so the measurements are read off the graph rather than kept twice.
    """
    world: WorldModel
    frames: list[int]                               # keyframes, sequence indices
    pairs: list[tuple[int, int]] = field(default_factory=list)       # odometry edges (anchor, j) in chain order
    candidates: list[tuple[int, int]] = field(default_factory=list)  # every closure proposed
    closures: list[tuple[int, int]] = field(default_factory=list)    # the candidates that passed every gate

    def _T_ji(self, edges) -> list[np.ndarray]:
        # the graph holds Z_ij = T_ij (WorldModel.measurement); the frontend's own output is T_ji
        return [e.measured_pose.inverse().matrix() for e in edges]

    @property
    def estimates(self) -> list[np.ndarray]:
        """T_ji per odometry pair."""
        return self._T_ji(self.world.factor_graph.edges[:len(self.pairs)])

    @property
    def closure_estimates(self) -> list[np.ndarray]:
        """T_ji per closure."""
        return self._T_ji(self.world.factor_graph.edges[len(self.pairs):])


@torch.no_grad()
def build(variant: Variant, sequence, model, on_edge: Callable[[Frontend], None] | None = None) -> Frontend:
    """
    Frontend only: encode keyframes, chain odometry edges, propose and verify
    closures. on_edge(front) is called after each odometry edge is added, so
    a viewer can grow with the chain.
    """
    cfg = variant.config
    # gap 2 on 0.6 m / 13.5 deg HM3D frames put keyframes 1.2 m and 27 deg
    # apart and bridged up to 117 of them (2026-09-30); such data is already at
    # keyframe spacing, so the gap applies to video-rate sequences only
    frames = keyframes(sequence, 1 if sequence.frames_are_keyframes else cfg.val_frame_gap, cfg.max_edges)

    # the C++ generator is shared, so without this the variants consume each
    # other's draws and reordering RUNS changes every number
    cpp.set_seed(cfg.seed)

    world = WorldModel(model, variant.pooling_fn, sequence, cfg, variant.edge_function)
    world.encode_keyframes(frames)      # batched; lazy per-frame encoding costs 2x
    front = Frontend(world, frames)

    # One edge at a time so the viewer grows with it. Bridge rather than
    # refuse: a refused odometry edge leaves its far frame without a vertex and
    # the trajectory forks into two gauge anchors.
    # Consecutive pairs are known up front, so match them batched. A bridged
    # keyframe makes the next pair (anchor, j + 2), which is matched singly.
    world.prefetch_matches(list(zip(frames[:-1], frames[1:])))
    # TODO: Reanchor after consecutive failures
    anchor = frames[0]
    for j in frames[1:]:
        if not world.add_edge(anchor, j):
            continue
        front.pairs.append((anchor, j))
        anchor = j
        if on_edge is not None:
            on_edge(front)

    front.candidates = world.loop_candidates(frames)
    world.prefetch_matches(front.candidates)
    for i, j in front.candidates:
        if world.add_edge(i, j, is_closure=True):
            front.closures.append((i, j))
    return front


def has_graph(traj_dir: Path) -> bool:
    return (traj_dir / GRAPH_FILE).exists()


def pooling_name(variant: Variant) -> str:
    return getattr(variant.pooling_fn, "__name__", repr(variant.pooling_fn))


def save_graph(traj_dir: Path, front: Frontend, variant: Variant, split: str | None = None) -> dict:
    """
    The unsolved graph as a Phase 3 sample, two files in traj_dir.

    graph.g2o, from FactorGraph.save_to_file, so the C++ loader, GTSAM and the
    trainer read one file: vertex ids are vertex indices, vertex 0 the gauge
    anchor at identity, edges in insertion order (odometry first, then
    closures), measurement Z_ij = T_ij (WorldModel.measurement), 17
    significant digits.

    graph.npz, what g2o cannot hold:

        frames            (K,)      keyframe sequence indices
        vertex_frames     (V,)      sequence index of each vertex, vertex order; bridged keyframes have none
        n_odometry        ()        edges [0, n_odometry) are odometry, the rest closures
        gt_poses          (V,4,4)   ground-truth T_WC of each vertex, the sequence's camera convention (OpenCV)
        candidates        (C,2)     every closure proposed, as frame pairs; those not among the
                                    closure edges were rejected by a gate
        config, variant, pooling, split, commit, created   strings

    Returns the summary written into meta.json.
    """
    graph = front.world.factor_graph
    order = sorted(front.world.vertex_index, key=front.world.vertex_index.get)
    sequence = front.world.sequence
    summary = dict(
        file=GRAPH_FILE, sidecar=GRAPH_SIDECAR, frames=len(front.frames), vertices=len(order),
        odometry_edges=len(front.pairs), closures=len(front.closures), candidates=len(front.candidates),
        variant=f"{solver_name(variant)} - {variant.config.run_name} - {pooling_name(variant)}",
        pooling=pooling_name(variant), split=split,
        commit=code_version(), created=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    graph.save_to_file(traj_dir / GRAPH_FILE)
    arrays = dict(
        frames=np.asarray(front.frames, dtype=np.int64),
        vertex_frames=np.asarray(order, dtype=np.int64),
        n_odometry=np.int64(len(front.pairs)),
        candidates=np.asarray(front.candidates, dtype=np.int64).reshape(-1, 2),
        config=np.array(json.dumps(asdict(variant.config))),
        variant=np.array(summary["variant"]),
        pooling=np.array(summary["pooling"]),
        split=np.array(str(split)),
        commit=np.array(summary["commit"]),
        created=np.array(summary["created"]),
    )
    if sequence.has_ground_truth:
        arrays["gt_poses"] = np.asarray([sequence.pose(f) for f in order], dtype=np.float64).reshape(-1, 4, 4)
    np.savez(traj_dir / GRAPH_SIDECAR, **arrays)
    return summary
