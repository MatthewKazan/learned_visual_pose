"""
Constructing DataLoaders, and feeding batches off them.

Both functions are about the same object, and neither belongs to training or to
evaluation -- both paths use them.
"""
from collections.abc import Iterator

import torch
from torch import Tensor
from torch.utils.data import DataLoader, ConcatDataset

from visual_pose.data_utils.wide_baseline import WideBaselineDataset
from visual_pose.config import Config
from visual_pose.data_utils.constants import DEVICE, REPO_DIR
from visual_pose.data_utils.dataset import TartanAirSequence
from visual_pose.data_utils.training_dataset import TACorrespondenceDataset


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


def build_wide_baseline_loaders(
        cfg: Config = Config()) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Train and val loaders mixing narrow and wide-baseline pairs.

    wide_to_narrow_ratio sets the mix. Wide pairs are the ones the descriptor
    fails on, so training needs them over-represented relative to their
    frequency in a sequence.
    """
    root = REPO_DIR / "data" / "tartan_air"
    common = dict(num_correspondences=cfg.num_correspondences,
                  sample_step=cfg.sample_step,
                  darkness_threshold=cfg.darkness_threshold,
                  max_depth=cfg.max_depth,
                  occlusion_tol=cfg.occlusion_tol)
    wide = dict(min_frame_gap=cfg.min_frame_gap,
                overlap_ratio_band=list(cfg.overlap_ratio_band))

    narrow_parts, wide_parts = [], []
    for name in cfg.train_sequences:
        sequence = TartanAirSequence(root / name)
        narrow = TACorrespondenceDataset(sequence, frame_gap=list(cfg.frame_gap),
                                         augment=True, jitter=cfg.jitter, **common)
        narrow_parts.append(narrow)
        # cap per sequence rather than globally, so one long sequence cannot
        # crowd out the others
        wide_parts.append(
            WideBaselineDataset(sequence, augment=True, jitter=cfg.jitter,
                                max_pairs=int(cfg.wide_to_narrow_ratio * len(narrow)),
                                seed=cfg.seed, **wide, **common))

    narrow_n = sum(len(d) for d in narrow_parts)
    wide_n = sum(len(d) for d in wide_parts)
    total = narrow_n + wide_n
    # the ratio is whatever the overlap band happened to yield, so report it --
    # a 95/5 split is a very different experiment from 50/50
    print(f"train pairs: {narrow_n} narrow + {wide_n} wide = {total} "
          f"({wide_n / max(total, 1):.0%} wide)")

    train_dataset = ConcatDataset(narrow_parts + wide_parts)

    val_sequence = TartanAirSequence(root / cfg.val_sequence)
    val_wide = WideBaselineDataset(val_sequence, eval=True,
                                   max_pairs=cfg.wide_val_pairs, seed=cfg.seed,
                                   **wide, **common)
    val_narrow = TACorrespondenceDataset(val_sequence, frame_gap=cfg.val_frame_gap,
                                         eval=True, **common)
    print(f"val pairs: {len(val_narrow)} narrow, {len(val_wide)} wide")

    # persistent_workers keeps workers alive across epochs. On macOS they are
    # spawned, not forked, so otherwise every epoch re-imports torch in each one.
    loader = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                  persistent_workers=cfg.num_workers > 0,
                  pin_memory=(DEVICE.type == "cuda"))
    return (DataLoader(train_dataset, shuffle=True, **loader),
            DataLoader(val_wide, shuffle=False, **loader),
            DataLoader(val_narrow, shuffle=False, **loader))


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
