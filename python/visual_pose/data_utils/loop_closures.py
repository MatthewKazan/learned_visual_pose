"""
Ground-truth loop closures of a sequence: every frame pair far enough apart in
time that sees the same surface, from depth and poses (generate_correspondences).
"""
from pathlib import Path

import numpy as np

from visual_pose.config import Config
from visual_pose.data_utils.constants import REPO_DIR
from visual_pose.data_utils.dataset import FrameSequence
from visual_pose.geometry.true_correspondences import generate_correspondences

CACHE_DIR = REPO_DIR / "data" / "cache"


class LoopClosures:
    def __init__(self, sequence: FrameSequence, config: Config, name: str | None = None,
                 min_overlap: float | None = None, regenerate_cache: bool = False):
        """
        name keys the cache under data/cache; None computes without caching.
        Pairs closer than config.min_frame_gap frames are never closures;
        min_overlap defaults to config.loop_true_overlap.
        """
        self.sequence = sequence
        self.cfg = config
        self.min_overlap = config.loop_true_overlap if min_overlap is None else min_overlap
        self.cache_path = self._cache_path(name) if name else None
        if self.cache_path and self.cache_path.exists() and not regenerate_cache:
            # print(f"loading overlaps from {self.cache_path}")
            self.overlap = np.load(self.cache_path)
        else:
            self.overlap = self.compute_overlap()
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                # via a temp file: a build killed mid-save left a 0-byte cache that loaded as EOFError
                tmp = self.cache_path.with_suffix(".tmp.npy")
                np.save(tmp, self.overlap)
                tmp.replace(self.cache_path)
        self.true_loop_closures = self.get_true_loop_closures()

    def compute_overlap(self) -> np.ndarray:
        """
        (N, N) float32: overlap[i, j] is the fraction of frame i's sample grid
        that lands visibly in frame j. Not symmetric (a close-up inside a wide
        shot is ~1 one way, small the other), so both directions are kept.
        0 for pairs under min_frame_gap. O(N^2) geometry passes, hence the cache.
        """
        seq, cfg = self.sequence, self.cfg
        n = len(seq)
        overlap = np.zeros((n, n), dtype=np.float32)
        # Loaded once: per pair, an HM3D depth PNG decode (5.8 ms) would cost
        # 140 s a trajectory. ~1.2 MB a frame, so 750 MB for TartanAir's longest.
        depths = [seq.depth(i) for i in range(n)]
        poses = [seq.pose(i) for i in range(n)]
        H, W = depths[0].shape
        n_grid = (H // cfg.sample_step) * (W // cfg.sample_step)
        for i in range(n):
            if i % 50 == 0:
                # prefixed: with several sequences in flight the lines interleave
                print(f"  {self.cache_path.stem if self.cache_path else 'overlap'}: frame {i}/{n}", flush=True)
            depth_i, T_Wi = depths[i], poses[i]
            for j in range(i + cfg.min_frame_gap, n):
                depth_j, T_Wj = depths[j], poses[j]
                for a, b, d_a, d_b, T_a, T_b in ((i, j, depth_i, depth_j, T_Wi, T_Wj),
                                                 (j, i, depth_j, depth_i, T_Wj, T_Wi)):
                    uv_a, _ = generate_correspondences(
                        depth_i=d_a, depth_j=d_b, T_Wi=T_a, T_Wj=T_b, K=seq.K,
                        sample_step=cfg.sample_step, grid_offset=(0, 0),
                        max_depth=cfg.max_depth, occlusion_tol=cfg.occlusion_tol)
                    overlap[a, b] = len(uv_a) / n_grid
        return overlap

    def get_true_loop_closures(self) -> list[tuple[int, int]]:
        """(i, j), i < j, where either frame sees more than min_overlap of itself in the other."""
        both = np.maximum(self.overlap, self.overlap.T)
        i, j = np.nonzero(np.triu(both > self.min_overlap))
        return list(zip(i.tolist(), j.tolist()))

    def in_band(self, low: float, high: float) -> list[tuple[int, int]]:
        """(i, j), i < j, with low < overlap[i, j] < high: i into j only, WideBaselineDataset's rule."""
        i, j = np.nonzero(np.triu((self.overlap > low) & (self.overlap < high)))
        return list(zip(i.tolist(), j.tolist()))

    def _cache_path(self, name: str) -> Path:
        c = self.cfg
        safe = name.replace("/", "_").replace(":", "_")
        return CACHE_DIR / f"loop_closures_{safe}_{c.min_frame_gap}_{c.sample_step}_{c.max_depth}_{c.occlusion_tol}.npy"
