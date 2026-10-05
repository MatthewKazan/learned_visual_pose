"""
Constructing DataLoaders, and feeding batches off them.

Both functions are about the same object, and neither belongs to training or to
evaluation -- both paths use them.
"""
from collections.abc import Iterator

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, ConcatDataset

from visual_pose.data_utils.wide_baseline_CNN import WideBaselineDataset
from visual_pose.config import Config
from visual_pose.data_utils.constants import DEVICE, REPO_DIR
from visual_pose.data_utils.dataset import FrameSequence, TartanAirSequence
from visual_pose.data_utils.training_dataset_CNN import TACorrespondenceDataset


def build_loaders(cfg: Config = Config()) -> tuple[DataLoader, DataLoader]:
    """(train_loader, val_loader). Batches are the dict TACorrespondenceDataset yields."""
    root = REPO_DIR / "data" / "tartan_air"
    common = dict(num_correspondences=cfg.num_correspondences,
                  sample_step=cfg.sample_step,
                  darkness_threshold=cfg.darkness_threshold,
                  max_depth=cfg.max_depth,
                  occlusion_tol=cfg.occlusion_tol)

    # augment on train only -- val has to stay a fixed yardstick
    train_dataset = ConcatDataset([
        TACorrespondenceDataset(TartanAirSequence(root / name),
                                frame_gap=list(cfg.frame_gap), augment=True,
                                jitter=cfg.jitter, **common)
        for name in cfg.train_sequences
    ])
    val_dataset = TACorrespondenceDataset(TartanAirSequence(root / cfg.val_sequence),
                                          frame_gap=cfg.val_frame_gap, eval=True, **common)

    # persistent_workers keeps workers alive across epochs. On macOS they are
    # spawned, not forked, so otherwise every epoch re-imports torch in each one.
    loader = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                  persistent_workers=cfg.num_workers > 0,
                  pin_memory=(DEVICE.type == "cuda"))
    return (DataLoader(train_dataset, shuffle=True, **loader),
            DataLoader(val_dataset, shuffle=False, **loader))


def wide_baseline_loaders(train: list[tuple[str, FrameSequence]], val: list[tuple[str, FrameSequence]],
                          cfg: Config, min_frame_gap: int, narrow_gaps: tuple[int, ...],
                          val_narrow_gap: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    (train, val_wide, val_narrow) loaders for any (cache name, sequence) lists.

    Train mixes narrow pairs (frames narrow_gaps apart) with wide ones (true
    closures at least min_frame_gap apart, in cfg.overlap_ratio_band), capped
    per sequence at wide_to_narrow_ratio x its narrow count so no sequence
    crowds out the rest. Wide pairs are the ones the descriptor fails on, so
    training needs them over-represented relative to their frequency. Val
    narrow is the tripwire: those are the pairs odometry runs on.
    """
    common = cfg.common()
    wide = dict(min_frame_gap=min_frame_gap, overlap_ratio_band=list(cfg.overlap_ratio_band))
    narrow_parts, wide_parts = [], []
    for name, seq in train:
        narrow = TACorrespondenceDataset(seq, frame_gap=list(narrow_gaps), augment=True,
                                         jitter=cfg.jitter, K=seq.K, **common)
        narrow_parts.append(narrow)
        wide_parts.append(WideBaselineDataset(seq, name=name, augment=True, jitter=cfg.jitter, K=seq.K,
                                              max_pairs=int(cfg.wide_to_narrow_ratio * len(narrow)),
                                              seed=cfg.seed, **wide, **common))
    val_wide = ConcatDataset([WideBaselineDataset(seq, name=name, eval=True, K=seq.K,
                                                  max_pairs=max(1, cfg.wide_val_pairs // len(val)),
                                                  seed=cfg.seed, **wide, **common) for name, seq in val])
    val_narrow = ConcatDataset([TACorrespondenceDataset(seq, frame_gap=val_narrow_gap, eval=True, K=seq.K, **common)
                                for _, seq in val])

    narrow_n, wide_n = sum(len(d) for d in narrow_parts), sum(len(d) for d in wide_parts)
    # the ratio is whatever the overlap band happened to yield, so report it --
    # a 95/5 split is a very different experiment from 50/50
    print(f"train pairs: {narrow_n} narrow + {wide_n} wide ({wide_n / max(narrow_n + wide_n, 1):.0%} wide); "
          f"val pairs: {len(val_narrow)} narrow, {len(val_wide)} wide")

    # persistent_workers keeps workers alive across epochs. On macOS they are
    # spawned, not forked, so otherwise every epoch re-imports torch in each one.
    loader = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                  persistent_workers=cfg.num_workers > 0,
                  pin_memory=(DEVICE.type == "cuda"))
    return (DataLoader(ConcatDataset(narrow_parts + wide_parts), shuffle=True, **loader),
            DataLoader(val_wide, shuffle=False, **loader),
            DataLoader(val_narrow, shuffle=False, **loader))


def build_wide_baseline_loaders(cfg: Config = Config()) -> tuple[DataLoader, DataLoader, DataLoader]:
    """TartanAir: cfg.train_sequences / cfg.val_sequence, video-rate frames."""
    root = REPO_DIR / "data" / "tartan_air"
    return wide_baseline_loaders([(n, TartanAirSequence(root / n)) for n in cfg.train_sequences],
                                 [(cfg.val_sequence, TartanAirSequence(root / cfg.val_sequence))], cfg,
                                 min_frame_gap=cfg.min_frame_gap, narrow_gaps=tuple(cfg.frame_gap),
                                 val_narrow_gap=cfg.val_frame_gap)


def build_hm3d_wide_baseline_loaders(cfg: Config = Config(), train_trajectories: int | None = None,
                                     val_trajectories: int = 20) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    HM3D, split by scene. Frames are keyframes 0.6 m apart, so narrow is the
    next keyframe and wide pairs are at least loop_min_gap apart -- what loop
    retrieval proposes. Reuses the overlap caches loop_closure_dataset built.
    train_trajectories evenly subsamples the 800 (None = all); val is
    trajectory_00 of up to val_trajectories held-out scenes.
    """
    from visual_pose.data_utils.loop_closure_dataset import trajectory_keys
    from visual_pose.data_utils.sources import source

    def evenly(keys: list[str], n: int | None) -> list[str]:
        return keys if n is None or n >= len(keys) else [keys[i] for i in np.linspace(0, len(keys) - 1, n).astype(int)]

    train = evenly(trajectory_keys("train"), train_trajectories)
    val = evenly([k for k in trajectory_keys("val") if k.endswith("trajectory_00")], val_trajectories)
    return wide_baseline_loaders([(f"{k}_kf1", source(k).frames()) for k in train],
                                 [(f"{k}_kf1", source(k).frames()) for k in val], cfg,
                                 min_frame_gap=cfg.loop_min_gap, narrow_gaps=(1,), val_narrow_gap=1)


def build_loop_closure_loaders(cfg: Config = Config()) -> tuple[DataLoader, DataLoader]:
    """(train_loader, val_loader) over LoopClosureDataset, split by scene. Needs its caches built."""
    from visual_pose.data_utils.loop_closure_dataset import LoopClosureDataset
    train = LoopClosureDataset("train", augment=True, jitter=cfg.jitter, seed=cfg.seed)
    # 16 per cell, not 64: 12.8k val pairs evaluate in ~5 min instead of ~20
    val = LoopClosureDataset("val", pairs_per_cell=16, seed=cfg.seed)
    loader = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                  persistent_workers=cfg.num_workers > 0,
                  pin_memory=(DEVICE.type == "cuda"))
    return (DataLoader(train, shuffle=True, **loader),
            DataLoader(val, shuffle=False, **loader))

def build_loop_closure_trajectory_loaders(cfg: Config = Config()) -> tuple[DataLoader, DataLoader]:
    """
    One trajectory per batch over LoopClosureTrajectories. Val is trajectory_00
    of each held-out scene: the trajectories experiments/loop_candidate_stats.py
    scores, so the checkpoint metric and that script agree.
    """
    from visual_pose.data_utils.loop_closure_dataset import LoopClosureTrajectories, trajectory_keys
    train = LoopClosureTrajectories("train", augment=cfg.aggregator_augment, jitter=cfg.jitter,
                                    stride=cfg.aggregator_frame_stride)
    val = LoopClosureTrajectories(keys=[k for k in trajectory_keys("val") if k.endswith("trajectory_00")])
    # 2 workers: an item is 60-75 MB of uint8 frames at stride 4 (~230 MB at
    # stride 1) and takes ~0.1 s off the SSD mirror; measured 2026-10-02, the
    # consumer waits 2.5 ms a batch against a 700 ms CNN pass, so 2 is plenty
    loader = dict(batch_size=1, num_workers=2, persistent_workers=True)
    return (DataLoader(train, shuffle=True, **loader),
            DataLoader(val, shuffle=False, **loader))


def build_loop_closure_probe_loader(cfg: Config = Config()) -> DataLoader:
    """
    trajectory_00 of the first cfg.aggregator_train_probe_scenes training scenes,
    every frame and no augmentation, scored exactly like val -- the train side
    of a train-vs-val comparison. These trajectories are also trained on.
    """
    from visual_pose.data_utils.loop_closure_dataset import LoopClosureTrajectories, trajectory_keys
    keys = [k for k in trajectory_keys("train") if k.endswith("trajectory_00")][:cfg.aggregator_train_probe_scenes]
    return DataLoader(LoopClosureTrajectories(keys=keys), batch_size=1, num_workers=2, persistent_workers=True)


def on_device(loader: DataLoader,
              device: torch.device) -> Iterator[dict[str, Tensor]]:
    """Wrap a DataLoader so every batch arrives on `device`.

    Workers are separate processes and always produce CPU tensors, so the move
    has to happen after collation.
    """
    for batch in loader:
        yield {
            k: v.to(device, non_blocking=True) if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }
