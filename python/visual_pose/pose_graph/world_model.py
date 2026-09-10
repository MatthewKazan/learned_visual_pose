"""
Incremental SLAM state: keyframes in, pose graph out.

Keyframes are cached, so F frames cost F encodes however many edges touch them.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from visual_pose import _geometry as cpp
from visual_pose.config import Config
from visual_pose.geometry.best_match import get_matching_pairs_from_descriptors
from visual_pose.geometry.true_correspondences import depth_at, usable_depth

# Below this a 3x3 sample covariance is mostly its own sampling noise (isotropic
# truth reaches condition 1000 at N=5), so edge_information drops to a scalar.
MIN_COVARIANCE_POINTS = 20
# 3N-6 degrees of freedom, so N=3 gives zero: no error estimate is possible.
MIN_VARIANCE_POINTS = 4


@dataclass
class KeyFrame:
    rgb: np.ndarray                # (H, W, 3) uint8, HWC as the loader gives it
    descriptors: Tensor            # (1, D, H', W'); stays a tensor for best_match
    global_descriptor: np.ndarray  # (D,) pooled fingerprint, for loop candidates


# Returns the fit, not just the pose, so the weight uses the same samples.
RelativePoseFn = Callable[[np.ndarray, np.ndarray, Config], cpp.RansacFit | None]

# Mutates graph.vertices and returns nothing; read the answer from poses().
# Bind solver keywords with partial, so cpp.gauss_newton needs no wrapper.
SolveFn = Callable[[cpp.FactorGraph], None]


class WorldModel:
    def __init__(self, descriptor_generator, pooling_method, sequence, cfg: Config,
                 get_relative_pose: RelativePoseFn, solve: SolveFn):
        self.descriptor_generator = descriptor_generator
        self.pooling_method = pooling_method
        self.factor_graph = cpp.FactorGraph()
        self.keyframes: dict[int, KeyFrame] = {}
        # sequence index -> vertex index; not all frames become keyframes
        self.vertex_index: dict[int, int] = {}
        self.sequence = sequence
        self.camera_model = cpp.PinholeCamera(self.sequence.K)
        self.cfg = cfg
        self.get_relative_pose = get_relative_pose
        self.solve = solve

    # -- keyframes ----------------------------------------------------------

    def create_keyframe(self, seq_idx: int) -> KeyFrame:
        """Encode one RGB frame: descriptor map plus pooled fingerprint."""
        rgb = self.sequence.rgb(seq_idx)
        image = (torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0).unsqueeze(0)
        image = image.to(self.descriptor_generator.device)
        descriptors, backbone = self.descriptor_generator(image)
        global_descriptor = self.pooling_method(backbone)[0]
        return KeyFrame(rgb, descriptors.detach(),
                        global_descriptor.detach().cpu().numpy())

    def keyframe(self, seq_idx: int) -> KeyFrame:
        """Cached create_keyframe."""
        if seq_idx not in self.keyframes:
            self.keyframes[seq_idx] = self.create_keyframe(seq_idx)
        return self.keyframes[seq_idx]

    # -- one pair -----------------------------------------------------------

    def correspondences(self, seq_idx_i: int, seq_idx_j: int):
        """
        (uv_i, uv_j, point_i, point_j) for one pair: pixels (K, 2) and metric
        points (K, 3) in each camera's own frame. K varies with the pair.
        """
        uv_i, uv_j = get_matching_pairs_from_descriptors(
            self.keyframe(seq_idx_i).descriptors,
            self.keyframe(seq_idx_j).descriptors,
            self.sequence.rgb(seq_idx_i).shape[:2],   # (H, W); rgb is HWC
            self.cfg.similarity_threshold,
        )
        # torch -> numpy boundary; float64 because the C++ takes MatrixX3d
        uv_i = uv_i.cpu().numpy().astype(np.float64)
        uv_j = uv_j.cpu().numpy().astype(np.float64)

        depth_i = depth_at(self.sequence.depth(seq_idx_i), uv_i)
        depth_j = depth_at(self.sequence.depth(seq_idx_j), uv_j)

        # not valid_depth_mask -- it re-indexes at integer uv and throws away
        # depth_at's subpixel interpolation
        mask = (usable_depth(depth_i, self.cfg.max_depth)
                & usable_depth(depth_j, self.cfg.max_depth))
        uv_i, uv_j = uv_i[mask], uv_j[mask]
        depth_i, depth_j = depth_i[mask], depth_j[mask]

        return (uv_i, uv_j,
                self.camera_model.back_project(uv_i, depth_i),
                self.camera_model.back_project(uv_j, depth_j))

    def loop_candidates(self, frames: list[int]) -> list[tuple[int, int]]:
        """
        Frame pairs at least loop_min_gap apart whose fingerprint cosine clears
        loop_retrieval_similarity. Detection only; add_edge verifies.
        """
        fingerprints = np.stack([self.keyframe(f).global_descriptor for f in frames])
        similarity = fingerprints @ fingerprints.T # F x F

        pairs = []
        for i in range(len(frames)):
            candidates = [j for j in range(i + 1, len(frames)) if abs(i - j) >= self.cfg.loop_min_gap]
            if candidates:
                pairs.extend([(i, j) for j in candidates if similarity[i, j] >= self.cfg.loop_retrieval_similarity])

        if self.cfg.loop_max_candidates and len(pairs) > self.cfg.loop_max_candidates:
            pairs = self._select(pairs, similarity, self.cfg.loop_max_candidates)
        return [(frames[i], frames[j]) for i, j in pairs]

    def _select(self, pairs, similarity, budget: int):
        """
        Spend a candidate budget across `pairs`. See Config.loop_selection.

        Seeded from cfg.seed: cpp.set_seed only reaches the C++ generator, so an
        unseeded numpy draw here makes the whole run unrepeatable.
        """
        mode = self.cfg.loop_selection
        if mode == "topk":
            return sorted(pairs, key=lambda ab: -similarity[ab])[:budget]
        if mode == "random":
            rng = np.random.default_rng(self.cfg.seed)
            return [pairs[k] for k in rng.permutation(len(pairs))[:budget]]
        if mode != "span":
            raise ValueError(f"unknown loop_selection {mode!r}, "
                             "expected topk|random|span")

        # Bands of equal span WIDTH, not equal population: the point is to give
        # under-populated spans a share the global ranking never gives them.
        spans = np.array([j - i for i, j in pairs])
        edges = np.linspace(spans.min(), spans.max() + 1, self.cfg.loop_span_bands + 1)
        band = np.digitize(spans, edges) - 1

        by_band = []
        for b in range(self.cfg.loop_span_bands):
            members = [pairs[k] for k in np.nonzero(band == b)[0]]
            by_band.append(sorted(members, key=lambda ab: -similarity[ab]))

        # Smallest bands first, so a band that cannot spend its share hands the
        # remainder to the others instead of leaving the budget unfilled.
        out, remaining = [], budget
        for n, ranked in enumerate(sorted(by_band, key=len)):
            share = remaining // (self.cfg.loop_span_bands - n)
            out.extend(ranked[:share])
            remaining -= min(share, len(ranked))
        return out

    # -- graph --------------------------------------------------------------

    def add_edge(self, seq_idx_i: int, seq_idx_j: int) -> bool:
        """
        Match a frame pair and add the constraint. False if unusable.

        A closure is an edge whose endpoints are both already vertices. Closures
        are gated on inlier count and on agreeing with the chain; odometry is
        not, because refusing it forks the trajectory into two gauge anchors.
        """
        is_closure = (seq_idx_i in self.vertex_index
                      and seq_idx_j in self.vertex_index)

        _uv_i, _uv_j, point_i, point_j = self.correspondences(seq_idx_i, seq_idx_j)
        fit = self.get_relative_pose(point_i, point_j, self.cfg)
        if fit is None:
            return False

        if is_closure:
            if len(fit.inliers) < self.cfg.loop_min_inliers:
                return False
            measured_tij = self.factor_graph.vertices[self.vertex_index[seq_idx_i]].pose.inverse() * self.factor_graph.vertices[self.vertex_index[seq_idx_j]].pose
            fit_diff = fit.T_ji * measured_tij
            if np.linalg.norm(fit_diff.translation()) > self.cfg.loop_max_dist:
                return False
        # before add_vertex: a late refusal would strand a vertex
        information = self.edge_information(point_i, point_j, fit)
        if information is None:
            return False
        # identity is the baseline the weighting is compared against
        if not self.cfg.weight_edges:
            information = np.eye(6)

        if seq_idx_i not in self.vertex_index and seq_idx_j not in self.vertex_index:
            # something has to fix the world frame; vertex 0 is the anchor
            self.vertex_index[seq_idx_i] = self.factor_graph.add_vertex(
                np.eye(4), float(seq_idx_i))

        # At most one fires. chain() wants T_known,new, so the two branches need
        # opposite subscript orders and exactly one inverts.
        if seq_idx_j not in self.vertex_index:
            self.vertex_index[seq_idx_j] = self.factor_graph.add_vertex(
                self.chain(seq_idx_i, fit.T_ji.inverse().matrix()), float(seq_idx_j))
        elif seq_idx_i not in self.vertex_index:
            self.vertex_index[seq_idx_i] = self.factor_graph.add_vertex(
                self.chain(seq_idx_j, fit.T_ji.matrix()), float(seq_idx_i))

        self.factor_graph.add_edge(
            self.vertex_index[seq_idx_i],
            self.vertex_index[seq_idx_j],
            self.measurement(fit),
            information,
        )
        return True

    def optimize(self) -> np.ndarray | None:
        """
        Calibrate the weights, solve, return (N, 3) positions.

        Below two vertices there is nothing to solve and the solver raises.
        """
        if len(self.factor_graph.vertices) < 2:
            return None
        self.calibrate()
        self.solve(self.factor_graph)
        return np.array(self.factor_graph.poses())[:, :3, 3]

    def calibrate(self) -> None:
        """
        Rescale every information matrix so a typical residual is 1.

        Does not move the optimum -- H and b scale together -- it only fixes
        what `huber` is compared against. Only non-zero residuals enter the
        median; the odometry chain reproduces its own measurements exactly.
        """
        edges, vertices = self.factor_graph.edges, self.factor_graph.vertices
        residuals = []
        for edge in edges:
            error = (edge.measured_pose.inverse()
                     * vertices[edge.from_index].pose.inverse()
                     * vertices[edge.to_index].pose)
            r = np.asarray(error.log()).ravel()
            residuals.append(np.sqrt(r @ edge.info_matrix @ r))

        nonzero = [r for r in residuals if r > 1e-9]
        if not nonzero:
            return
        median = float(np.median(nonzero))
        self.factor_graph.set_information(
            [edge.info_matrix / median ** 2 for edge in edges])

    def measurement(self, fit: cpp.RansacFit) -> np.ndarray:
        """
        (4, 4) edge measurement Z_ij = T_ij, the inverse of the frontend's T_ji.

        Backwards still converges, to the wrong answer. The test that separates
        the two seeds vertices from ground truth rather than from chain().
        """
        return fit.T_ji.inverse().matrix()

    def chain(self, known: int, T_known_new: np.ndarray) -> np.ndarray:
        """
        (4, 4) starting absolute pose for a new vertex:

            T_W,new = T_W,known @ T_known,new

        `known` is a sequence index; the caller supplies that subscript order.
        """
        T_W_known = self.factor_graph.vertices[self.vertex_index[known]].pose
        return T_W_known.matrix() @ T_known_new

    def edge_information(self, P_i: np.ndarray, P_j: np.ndarray, fit):
        """
        (6, 6) information for one edge, or None if the fit cannot estimate its
        own error.

            Omega = sum_k J_k^T Sigma_p^-1 J_k,   J_k = [-I, hat(q_k)]

        Sigma_p carries the noise scale, so no separate sigma^2. Point count
        enters through the sum, point layout through hat(q_k) -- a flat cloud
        leaves the rotation about its normal weakly weighted.
        """
        if len(fit.inliers) < MIN_VARIANCE_POINTS:
            return None

        P_i = P_i[fit.inliers]
        P_j = P_j[fit.inliers]

        q = P_i @ fit.T_ji.rotation().matrix().T + fit.T_ji.translation()

        residual = P_j - q

        # Weak, not absent. The floor guards the 3x3 only; one variance needs
        # far fewer points, so the scalar stays usable below it.
        if len(fit.inliers) < MIN_COVARIANCE_POINTS:
            variance = (residual ** 2).sum() / (3 * len(residual) - 6)
            return np.eye(6) * (len(residual) / variance)

        covariance = residual.T @ residual / (len(residual) - 2)
        try:
            point_information = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            # residuals confined to a plane or a line -- depth-only error does
            # this, and it is a real configuration, not a numerical accident
            return None

        pose_information = np.zeros((6, 6))

        for q_k in q:
            J = np.zeros((3, 6))
            J[:, :3] = -np.eye(3)
            J[:, 3:] = cpp.hat(q_k)

            pose_information += J.T @ point_information @ J

        return pose_information

