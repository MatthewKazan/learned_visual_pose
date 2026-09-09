from multiprocessing import Pool, cpu_count
from pathlib import Path

import torch

from visual_pose.data_utils.dataset import TartanAirSequence
from visual_pose.config import Config
from visual_pose.geometry.true_correspondences import generate_correspondences
from visual_pose.data_utils.training_dataset import TACorrespondenceDataset
from visual_pose.data_utils.constants import REPO_DIR

class WideBaselineDataset(TACorrespondenceDataset):

    def __init__(
            self,
            *args,
            min_frame_gap: int = 60,
            overlap_ratio_band: list[float] | None = None,
            regenerate_cache: bool = False,
            max_pairs: int = 16000,
            seed: int = 0,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.overlap_ratio_band = [0.1, 0.6] if overlap_ratio_band is None else overlap_ratio_band

        self.min_frame_gap = min_frame_gap
        self.cache_path = self._cache_path()

        self.pairs = []
        if not self.cache_path.exists() or regenerate_cache:
            print(f"generating pairs to {self.cache_path}")
            self.pairs = self.generate_pairs()
        else:
            print(f"loading pairs from {self.cache_path}")
            self.pairs = torch.load(self.cache_path)

        # Capped after the load so the cache stays complete and changing
        # max_pairs costs nothing. Seeded so two runs see the same subset.
        if max_pairs is not None and len(self.pairs) > max_pairs:
            generator = torch.Generator().manual_seed(seed)
            keep = torch.randperm(len(self.pairs), generator=generator)[:max_pairs]
            self.pairs = [self.pairs[i] for i in sorted(keep.tolist())]

    def generate_pairs(self) -> list[tuple[int, int]]:
        """
        Every (i, j) whose ground-truth overlap falls in the band. O(n^2) in
        sequence length with a full geometry pass per candidate, hence the cache.
        """
        pairs = []
        n_grid = (self.seq.depth(0).shape[0] // self.sample_step) * (self.seq.depth(0).shape[1] // self.sample_step)

        for i in range(len(self.seq)):
            if i % 50 == 0:
                # prefixed: with several sequences in flight the lines interleave
                print(f"  {self.seq.dataset_dir.name}: frame {i}/{len(self.seq)}, "
                      f"{len(pairs)} pairs so far", flush=True)
                print(f"pairs seen so far: {len(pairs)}")
            for j in range(i + self.min_frame_gap, len(self.seq)):

                grid_offset = (0, 0)

                uv_i, uv_j = generate_correspondences(
                    depth_i=self.seq.depth(i),
                    depth_j=self.seq.depth(j),
                    T_Wi=self.seq.pose(i),
                    T_Wj=self.seq.pose(j),
                    K=self.K,
                    sample_step=self.sample_step,
                    grid_offset=grid_offset,
                    max_depth=self.max_depth,
                    occlusion_tol=self.occlusion_tol,
                )
                overlap_ratio = len(uv_i) / n_grid

                if self.overlap_ratio_band[0] < overlap_ratio < self.overlap_ratio_band[1]:
                    pairs.append((i, j))
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(pairs, self.cache_path)
        return pairs

    def _cache_path(self) -> Path:
        filename = f"wide_baseline_cache_{self.seq.dataset_dir.name}_{self.min_frame_gap}_{self.overlap_ratio_band[0]}_{self.overlap_ratio_band[1]}_{self.sample_step}_{self.max_depth}_{self.occlusion_tol}.pt"
        return self.seq.dataset_dir.parent.parent / "cache" / filename

def build_one(name: str) -> tuple[str, int]:
    """
    One sequence, start to finish. Module level so it is picklable -- macOS
    spawns rather than forks, so a closure or a lambda cannot be the worker.
    """
    cfg = Config()
    sequence = TartanAirSequence(REPO_DIR / "data" / "tartan_air" / name)
    dataset = WideBaselineDataset(
        sequence,
        min_frame_gap=60,
        overlap_ratio_band=[0.1, 0.6],
        eval=True,
        frame_gap=5,
        **cfg.common(),
    )
    return name, len(dataset)


if __name__ == "__main__":
    cfg = Config()
    # dict.fromkeys dedupes while keeping order, in case val is also in train
    names = list(dict.fromkeys(list(cfg.train_sequences) + [cfg.val_sequence]))

    # Processes, not threads: PNG decode and a Python loop, neither of which
    # releases the GIL. Two cores left free to keep the machine usable.
    workers = max(1, min(len(names), cpu_count() - 2))
    print(f"scanning {len(names)} sequences on {workers} processes: {', '.join(names)}")

    with Pool(processes=workers) as pool:
        for name, count in pool.imap_unordered(build_one, names):
            print(f"DONE {name}: {count} pairs", flush=True)
