from dataclasses import replace
from multiprocessing import Pool, cpu_count

import torch

from visual_pose.data_utils.dataset import TartanAirSequence
from visual_pose.config import Config
from visual_pose.data_utils.loop_closures import LoopClosures
from visual_pose.data_utils.training_dataset_CNN import TACorrespondenceDataset
from visual_pose.data_utils.constants import REPO_DIR


class WideBaselineDataset(TACorrespondenceDataset):
    """
    The sequence's true loop closures (LoopClosures), narrowed to the overlap
    band worth training on. Any FrameSequence: `name` keys the overlap cache
    (default the TartanAir directory name; HM3D passes f"{key}_kf1", the cache
    loop_closure_dataset already built). `min_frame_gap` counts the sequence's
    own frames: 60 at TartanAir's video rate, loop_min_gap (50) for HM3D keyframes.
    """

    def __init__(
            self,
            *args,
            name: str | None = None,
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

        cfg = replace(Config(), min_frame_gap=min_frame_gap, sample_step=self.sample_step,
                      max_depth=self.max_depth, occlusion_tol=self.occlusion_tol)
        self.loop_closures = LoopClosures(self.seq, cfg, name=name or self.seq.dataset_dir.name,
                                          regenerate_cache=regenerate_cache)
        self.pairs = self.loop_closures.in_band(*self.overlap_ratio_band)

        # Capped after the load so the cache stays complete and changing
        # max_pairs costs nothing. Seeded so two runs see the same subset.
        if max_pairs is not None and len(self.pairs) > max_pairs:
            generator = torch.Generator().manual_seed(seed)
            keep = torch.randperm(len(self.pairs), generator=generator)[:max_pairs]
            self.pairs = [self.pairs[i] for i in sorted(keep.tolist())]


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
