"""
Checkpoint round-trip. Imported by training (to save) and by every evaluation
script (to load), so it belongs to neither.
"""
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from visual_pose.config import Config
from visual_pose.data_utils.constants import DEVICE
from visual_pose.models.descriptor_cnn import DescriptorCNN

def save_checkpoint(path: Path, model: nn.Module, optimizer: Optimizer,
                    lr_scheduler: LRScheduler, epoch: int,
                    val_mma: float) -> None:
    """
    Save enough state to *resume*, not just to run inference: optimizer momentum
    and scheduler position matter as much as the weights.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "val_mma": val_mma,
    }, path)


def load_model(cfg: Config, checkpoint: str = "best.pt") -> DescriptorCNN:
    """Architecture from cfg, weights from cfg.checkpoint_dir / checkpoint."""
    model = DescriptorCNN.from_config(cfg).to(DEVICE)

    state = torch.load(cfg.checkpoint_dir / checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model"])
    return model
