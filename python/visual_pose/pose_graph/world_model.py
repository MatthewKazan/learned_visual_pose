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
from visual_pose.geometry.best_match import get_matching_pairs_batched
from visual_pose.geometry.true_correspondences import (
    back_project, depth_at, in_image_mask, project, sample_pixel_grid, transform_points,
    usable_depth, valid_depth_mask)

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
                 get_relative_pose: RelativePoseFn):
        self.descriptor_generator = descriptor_generator
        self.pooling_method = pooling_method
        self.factor_graph = cpp.FactorGraph()
        self.keyframes: dict[int, KeyFrame] = {}
        # depth maps by frame: TartanAir loads them from disk and a run touches
        # each keyframe's map ~6 times (both ends of every pair, plus the gate)
        self._depth: dict[int, np.ndarray] = {}
        # pixel matches by pair, filled by prefetch_matches (batched on the
        # GPU) or on demand one pair at a time
        self._matches: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        # sequence index -> vertex index; not all frames become keyframes
        self.vertex_index: dict[int, int] = {}
        # The odometry chain, for gating closures against accumulated drift.
        # chain_links: (from frame, to frame, edge covariance in frame `to`) per
        # odometry edge in order; chain_prefix[k]: world-frame covariance of
        # the vertex at chain position k, accumulated from the anchor;
        # chain_pos: frame -> chain position. Indexed by chain position, not
        # vertex index: a closure candidate whose endpoint was bridged appends
        # a vertex out of frame order.
        self.chain_links: list[tuple[int, int, np.ndarray]] = []
        self.chain_prefix: list[np.ndarray] = []
        self.chain_pos: dict[int, int] = {}
        self.sequence = sequence
        self.camera_model = cpp.PinholeCamera(self.sequence.K)
        self.cfg = cfg
        self.get_relative_pose = get_relative_pose
        self.loop_closure_clusters = []

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

    @torch.no_grad()
    def encode_keyframes(self, frames: list[int], batch_size: int = 16) -> None:
        """
        Fill the keyframe cache for `frames` in batches. Same result as one
        create_keyframe per frame (eval-mode BatchNorm, per-sample pooling);
        measured 11.8 ms/frame one at a time vs 5.5 ms in batches of 16.
        """
        todo = [f for f in frames if f not in self.keyframes]
        for start in range(0, len(todo), batch_size):
            chunk = todo[start:start + batch_size]
            rgbs = [self.sequence.rgb(f) for f in chunk]
            images = torch.from_numpy(np.stack(rgbs)).permute(0, 3, 1, 2).float().div_(255.0)
            descriptors, backbone = self.descriptor_generator(images.to(self.descriptor_generator.device))
            fingerprints = self.pooling_method(backbone).detach().cpu().numpy()
            for k, f in enumerate(chunk):
                self.keyframes[f] = KeyFrame(rgbs[k], descriptors[k:k + 1].detach(), fingerprints[k])

    def depth(self, seq_idx: int) -> np.ndarray:
        """Cached sequence.depth. For in-memory sequences this stores a reference, so it costs nothing."""
        if seq_idx not in self._depth:
            self._depth[seq_idx] = self.sequence.depth(seq_idx)
        return self._depth[seq_idx]

    # -- one pair -----------------------------------------------------------

    @torch.no_grad()
    def prefetch_matches(self, pairs: list[tuple[int, int]], batch_size: int = 4) -> None:
        """
        Match `pairs` on the GPU `batch_size` at a time and cache the pixel
        matches for correspondences(). Per pair on 200 real P006 pairs, MPS:
        9.8 ms singly, 5.9 / 5.7 / 6.8 / 7.8 ms at batch 2 / 4 / 8 / 16 -- the
        (B, N, M) similarity is 92 MB per pair and past 4 it costs more than
        it saves. A pair that was not prefetched is matched on its own when
        asked for, so a partial prefetch costs time, never results.
        """
        todo = [p for p in pairs if p not in self._matches]
        for start in range(0, len(todo), batch_size):
            chunk = todo[start:start + batch_size]
            descriptors_i = torch.cat([self.keyframe(i).descriptors for i, _ in chunk])
            descriptors_j = torch.cat([self.keyframe(j).descriptors for _, j in chunk])
            image_shape = self.keyframe(chunk[0][0]).rgb.shape[:2]     # (H, W); rgb is HWC
            matched = get_matching_pairs_batched(descriptors_i, descriptors_j, image_shape,
                                                 self.cfg.similarity_threshold)
            for pair, (uv_i, uv_j) in zip(chunk, matched):
                # torch -> numpy boundary; float64 because the C++ takes MatrixX3d
                self._matches[pair] = (uv_i.cpu().numpy().astype(np.float64),
                                       uv_j.cpu().numpy().astype(np.float64))

    def matches(self, seq_idx_i: int, seq_idx_j: int) -> tuple[np.ndarray, np.ndarray]:
        """(uv_i, uv_j) pixel matches, (K, 2) each; from the cache or matched now."""
        if (seq_idx_i, seq_idx_j) not in self._matches:
            self.prefetch_matches([(seq_idx_i, seq_idx_j)])
        return self._matches[(seq_idx_i, seq_idx_j)]

    def correspondences(self, seq_idx_i: int, seq_idx_j: int):
        """
        (uv_i, uv_j, point_i, point_j) for one pair: pixels (K, 2) and metric
        points (K, 3) in each camera's own frame. K varies with the pair.
        """
        uv_i, uv_j = self.matches(seq_idx_i, seq_idx_j)

        depth_i = depth_at(self.depth(seq_idx_i), uv_i)
        depth_j = depth_at(self.depth(seq_idx_j), uv_j)

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
            candidates = [j for j in range(i + 1, len(frames)) if j - i >= self.cfg.loop_min_gap]
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

    def add_edge(self, seq_idx_i: int, seq_idx_j: int, *, is_closure: bool = False) -> bool:
        """
        Match a frame pair and add the constraint. False if unusable.

        Odometry (is_closure=False) extends the chain: i is the anchor, j becomes a
        vertex. It is not refused for few inliers, because that forks the
        trajectory into two gauge anchors; it gets an unknown covariance instead.
        A closure (is_closure=True) joins two existing vertices, gated on inlier
        count and on the chain-consistency test. A candidate touching a
        keyframe the chain never reached is refused: vertices enter the graph
        through odometry only. Letting candidates add them hung bridged
        keyframes off single unverified edges (ATE 1.29 m against 0.10 m for
        the chain) and, when both ends were bridged, planted extra anchors at
        the origin (00800 trajectory_01, 2026-09-29).
        """
        known_i, known_j = seq_idx_i in self.vertex_index, seq_idx_j in self.vertex_index
        if is_closure and not (known_i and known_j):
            return False
        if not is_closure and (known_j or (not known_i and self.factor_graph.vertices)):
            return False    # odometry must run anchor -> new frame

        _uv_i, _uv_j, point_i, point_j = self.correspondences(seq_idx_i, seq_idx_j)
        fit = self.get_relative_pose(point_i, point_j, self.cfg)
        if fit is None:
            return False
        if not is_closure and fit.T_ji.rotation().magnitude() > self.cfg.odometry_max_rotation_rad:
            return False

        # before add_vertex: a late refusal would strand a vertex
        information = self.edge_information(point_i, point_j, fit, is_closure)
        if information is None:
            return False
        # Odometry is not refused for few inliers (see docstring), but its
        # covariance then comes from Config, not from residuals RANSAC chose.
        if not is_closure and len(fit.inliers) < self.cfg.odometry_min_inliers:
            information = self.unknown_information()
        # identity is the baseline the weighting is compared against
        if not self.cfg.weight_edges:
            information = np.eye(6)

        measurement = self.measurement(fit)

        if is_closure:
            if len(fit.inliers) < self.cfg.loop_min_inliers:
                return False
            if self.depth_agreement(seq_idx_i, seq_idx_j, measurement) < self.cfg.closure_min_agreement:
                return False
            if not self.chain_consistent_gate(seq_idx_i, seq_idx_j, measurement, information):
                return False

        if seq_idx_i not in self.vertex_index and seq_idx_j not in self.vertex_index:
            # something has to fix the world frame; vertex 0 is the anchor
            self.vertex_index[seq_idx_i] = self.factor_graph.add_vertex(
                np.eye(4), float(seq_idx_i))
            self.chain_pos[seq_idx_i] = 0
            self.chain_prefix = [np.zeros((6, 6))]

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
            measurement,
            information,
        )
        if not is_closure:
            self.chain_links.append((seq_idx_i, seq_idx_j, np.linalg.inv(information)))
            self.chain_pos[seq_idx_j] = len(self.chain_prefix)
            self.chain_prefix.append(self.chain_prefix[self.chain_pos[seq_idx_i]]
                                     + self._to_world(seq_idx_j, self.chain_links[-1][2]))
        else:
            # Joins the cluster whose SEED (first closure) it is near, not any
            # member: matching any member chains a whole twice-walked corridor
            # into one cluster, and 1/size would then gut a long revisit whose
            # closures constrain different parts of the trajectory.
            r = self.cfg.loop_closure_range
            for cluster in self.loop_closure_clusters:
                i, j = cluster[0]
                if abs(seq_idx_i - i) < r and abs(seq_idx_j - j) < r:
                    cluster.append((seq_idx_i, seq_idx_j))
                    break
            else:
                self.loop_closure_clusters.append([(seq_idx_i, seq_idx_j)])
        return True

# -- closure gates --------------------------------------------------------

    def depth_agreement(self, i: int, j: int, T_ij: np.ndarray, stride: int = 8) -> float:
        """
        Fraction of frame i's grid pixels whose depth, transformed by T_ij into
        frame j and projected, agrees with frame j's depth there (relative
        tolerance Config.closure_depth_tolerance). Counted over pixels that land
        in front of camera j, inside its image, on valid depth.

        A correct transform explains the whole overlap; a fit on one flat patch
        explains only that patch. See Config.closure_depth_tolerance for the numbers.
        """
        depth_i, depth_j = self.depth(i), self.depth(j)
        H, W = depth_i.shape
        uv = sample_pixel_grid(H, W, stride)
        uv = uv[valid_depth_mask(depth_i, uv, self.cfg.max_depth)]
        P_i = back_project(uv.astype(np.float64), depth_i[uv[:, 1], uv[:, 0]], self.sequence.K)
        P_j = transform_points(P_i, np.linalg.inv(T_ij))       # T_ij maps frame j into i
        P_j = P_j[P_j[:, 2] > 0.05]
        if len(P_j) < 50:
            return 0.0
        uv_j = project(P_j, self.sequence.K)
        inside = in_image_mask(uv_j, H, W)
        if inside.sum() < 50:
            return 0.0
        predicted = P_j[inside][:, 2]
        measured = depth_at(depth_j, uv_j[inside])
        valid = usable_depth(measured, self.cfg.max_depth)
        if valid.sum() < 50:
            return 0.0
        agree = np.abs(measured[valid] - predicted[valid]) < self.cfg.closure_depth_tolerance * predicted[valid]
        return float(agree.mean())

    def unknown_information(self) -> np.ndarray:
        """Information of an edge we could not verify: diagonal, translation then rotation, 1/sigma^2. Positive definite, so 'unknown' is a large sigma, never zero."""
        sigma = np.array([self.cfg.unknown_edge_sigma_t] * 3 + [self.cfg.unknown_edge_sigma_r] * 3)
        return np.diag(1.0 / sigma ** 2)

    def _to_world(self, frame: int, covariance: np.ndarray) -> np.ndarray:
        """
        Edge covariance from frame `frame` into the world frame: Ad Sigma Ad^T
        with Ad the adjoint of that vertex's pose.

        edge_information's J = [-I, hat(q)] perturbs T_ji on the left with q in
        frame j, so its covariance is a frame-j perturbation. With T_Wj = T_Wi Z
        and Z = T_ji^-1 that perturbation lands on the right of T_Wj, and
        T Exp(d) = Exp(Ad_T d) T moves it to the world side.
        """
        Ad = np.asarray(self.factor_graph.vertices[self.vertex_index[frame]].pose.adjoint())
        return Ad @ covariance @ Ad.T

    def rebuild_chain_covariance(self) -> None:
        """
        After a solve: the transports depend on the vertex poses, the stored
        per-edge covariances do not.

        chain_links holds RAW covariances and the gate uses raw closure
        information on purpose: calibrate() rescales the graph's matrices by a
        per-run factor for Huber's sake, and the gate needs absolute scale.
        """
        if not self.chain_prefix:
            return
        self.chain_prefix = [np.zeros((6, 6))]
        for a, b, cov in self.chain_links:
            self.chain_prefix.append(self.chain_prefix[self.chain_pos[a]] + self._to_world(b, cov))

    def chain_consistent_gate(self, i: int, j: int, Z_ij: np.ndarray, information: np.ndarray) -> bool:
        """
        Mahalanobis test of a closure against the odometry drift between its endpoints.

        Predict j from i through the closure and compare with vertex j:

            r = Log(T_Wi Z_ij T_Wj^-1)

        a left perturbation of T_Wj in the world frame. Its covariance is the
        chain's accumulated world-frame covariance between the two chain
        positions plus the closure's own, transported the same way. Under an
        independent-error model d^2 ~ chi^2(6); see Config.closure_gate_threshold
        for why the threshold is not that quantile.

        Verified 2026-09-29: the residual computed here matched the actual chain
        drift against ground truth on every closure of P006, P000 and
        trajectory_20. A bad odometry edge carries a large covariance (see
        Config.odometry_min_inliers), so the closures crossing it are exactly
        the ones this test lets through.
        """
        if i not in self.chain_pos or j not in self.chain_pos:
            return True     # off-chain endpoint: nothing to test against
        vertices = self.factor_graph.vertices
        T_Wi, T_Wj = vertices[self.vertex_index[i]].pose, vertices[self.vertex_index[j]].pose
        r = np.asarray((T_Wi * cpp.PoseSE3(Z_ij) * T_Wj.inverse()).log()).ravel()
        if np.linalg.norm(r[3:]) > self.cfg.max_residual_threshold:
            return False
        # A fit that collapses the baseline: measured translation a small
        # fraction of what the chain says, over a chain span long enough that
        # drift cannot explain it. See Config.closure_collapse_ratio.
        t_chain = np.linalg.norm(np.asarray((T_Wi.inverse() * T_Wj).translation()))
        t_meas = np.linalg.norm(np.asarray(Z_ij)[:3, 3])
        if t_chain > self.cfg.closure_collapse_min_m and t_meas < self.cfg.closure_collapse_ratio * t_chain:
            return False
        lo, hi = sorted((self.chain_pos[i], self.chain_pos[j]))
        covariance = (self.chain_prefix[hi] - self.chain_prefix[lo]
                      + self._to_world(j, np.linalg.inv(information)))
        d_squared = float(r @ np.linalg.solve(covariance, r))
        return d_squared <= self.cfg.closure_gate_threshold

    def optimize(self, solve: SolveFn) -> np.ndarray | None:
        """
        Calibrate the weights, solve, return (N, 3) positions.

        Below two vertices there is nothing to solve and the solver raises.
        """
        if len(self.factor_graph.vertices) < 2:
            return None
        self.calibrate()
        solve(self.factor_graph)
        self.rebuild_chain_covariance()
        return np.array(self.factor_graph.poses())[:, :3, 3]

    def calibrate(self) -> None:
        """
        Rescale every information matrix so a typical residual is 1.

        Does not move the optimum -- H and b scale together -- it only fixes
        what `huber` is compared against. Only non-zero residuals enter the
        median; the odometry chain reproduces its own measurements exactly.
        Touches the graph only: the closure gate keeps its own raw copies
        (rebuild_chain_covariance) because this factor is per run.
        """
        # One revisit's closures are near-copies of one measurement; summed as
        # independent they over-trust it (00870, 00895). Each member of a
        # cluster gets 1/size, so the cluster weighs about one closure.
        # Clusters hold frame numbers, edges hold vertex numbers: they differ
        # after the first bridged keyframe, so map through vertex_index.
        info_scale = {}
        for cluster in self.loop_closure_clusters:
            for i, j in cluster:
                vi, vj = self.vertex_index[i], self.vertex_index[j]
                info_scale[(vi, vj)] = info_scale[(vj, vi)] = 1 / len(cluster)
        edges, vertices = self.factor_graph.edges, self.factor_graph.vertices
        residuals = []
        info_matrices = []
        for edge in edges:
            information = edge.info_matrix * info_scale.get((edge.from_index, edge.to_index), 1.0)
            error = (edge.measured_pose.inverse()
                     * vertices[edge.from_index].pose.inverse()
                     * vertices[edge.to_index].pose)
            r = np.asarray(error.log()).ravel()
            residuals.append(np.sqrt(r @ information @ r))     # of the graph actually solved
            info_matrices.append(information)

        nonzero = [r for r in residuals if r > 1e-9]
        if not nonzero:
            return
        median = float(np.median(nonzero))
        self.factor_graph.set_information(
            [mat / median ** 2 for mat in info_matrices])



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

    def edge_information(self, P_i: np.ndarray, P_j: np.ndarray, fit, loop_closure: bool):
        """
        (6, 6) POSE information for one edge, or None if the fit cannot estimate its
        own error.

            Omega = (N_eff / N) sum_k J_k^T Sigma_p^-1 J_k,   J_k = [-I, hat(q_k)]

        Sigma_p carries the noise scale, so no separate sigma^2. Point layout
        enters through hat(q_k) -- a flat cloud leaves the rotation about its
        normal weakly weighted. Point count does NOT: the sum alone made
        confidence grow linearly with N against ground truth, so N_eff
        (Config.edge_effective_inliers) caps it.
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
            return np.eye(6) * (self.cfg.edge_effective_inliers / variance)

        covariance = residual.T @ residual / (len(residual) - 2)
        eigvals = np.linalg.eigvalsh(covariance)
        if loop_closure and np.max(eigvals) / np.min(eigvals) > self.cfg.info_ratio_threshold:
            return None
        # Floor each principal variance: inlier residuals on a plane put sigma
        # 0.001-0.007 mm on its normal, against 0.14-1.7 mm (p5) on true pairs,
        # and inverting that gave one edge 99% of a graph's cost. See
        # Config.point_sigma_floor_m.
        # TODO: UNDERSTAND THIS
        w, basis = np.linalg.eigh(covariance)
        covariance = (basis * np.maximum(w, self.cfg.point_sigma_floor_m ** 2)) @ basis.T
        try:
            point_information = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            # residuals confined to a plane or a line -- depth-only error does
            # this, and it is a real configuration, not a numerical accident
            return None

        # sum_k J_k^T S J_k with J_k = [-I, hat(q_k)] and S = point_information, in blocks:
        #   tt = N S                       tr = -S sum_k hat(q_k) = -S hat(sum_k q_k)
        #   rr = sum_k hat(q_k)^T S hat(q_k)   -- a sum of products, the only per-point term
        # Was a Python loop with one C++ hat() call per inlier: 7.3 ms per edge at
        # N=1800, a fifth of a run; this is ~0.3 ms and agrees to 1e-9.
        n = len(q)
        hat_q = np.zeros((n, 3, 3))
        hat_q[:, 0, 1], hat_q[:, 0, 2] = -q[:, 2], q[:, 1]
        hat_q[:, 1, 0], hat_q[:, 1, 2] = q[:, 2], -q[:, 0]
        hat_q[:, 2, 0], hat_q[:, 2, 1] = -q[:, 1], q[:, 0]
        hat_sum = cpp.hat(q.sum(axis=0))

        pose_information = np.zeros((6, 6))
        pose_information[:3, :3] = n * point_information
        pose_information[:3, 3:] = -point_information @ hat_sum
        pose_information[3:, :3] = pose_information[:3, 3:].T
        pose_information[3:, 3:] = np.einsum("nji,jk,nkl->il", hat_q, point_information, hat_q)

        return pose_information * (self.cfg.edge_effective_inliers / n)
