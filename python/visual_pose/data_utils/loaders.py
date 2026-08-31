"""
Constructing DataLoaders, and feeding batches off them.

Both functions are about the same object, and neither belongs to training or to
evaluation -- both paths use them.
"""
import torch
from torch import Tensor
from torch.utils.data import DataLoader, ConcatDataset

from visual_pose.config import Config
from visual_pose.data_utils.constants import DEVICE, REPO_DIR
from visual_pose.data_utils.dataset import TartanAirSequence
from visual_pose.data_utils.training_dataset import TACorrespondenceDataset


def build_loaders(cfg: Config):
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


def on_device(loader, device):
    """
    Wrap a DataLoader so every batch arrives on `device`.

    DataLoader workers are separate processes and always produce CPU tensors,
    so the move has to happen after collation. Doing it here rather than in
    each loop body means it can't be forgotten in one loop and not another.
    """
    for batch in loader:
        yield {
            k: v.to(device, non_blocking=True) if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }
