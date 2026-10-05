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
from visual_pose.data_utils.constants import DEVICE, REPO_DIR
from visual_pose.models.descriptor_cnn import DescriptorCNN

def save_checkpoint(path: Path, model: nn.Module, optimizer: Optimizer,
                    lr_scheduler: LRScheduler, epoch: int,
                    val_metric: float, metric_name: str = "val metric") -> None:
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
        "val_metric": val_metric,
        "metric_name": metric_name,
    }, path)


def load_cnn_model(cfg: Config, checkpoint: str = "best.pt") -> DescriptorCNN:
    """Architecture from cfg, weights from cfg.checkpoint_dir / checkpoint."""
    model = DescriptorCNN.from_config(cfg).to(DEVICE)

    state = torch.load(cfg.checkpoint_dir / checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model"])
    return model


def load_learned_pooling(run_name: str = "attention_pooling_infonce", input_dim: int = 256,
                         checkpoint: str = "best.pt"):
    """
    A trained AttentionPooling as a WorldModel pooling_method: (B, 256, H', W')
    backbone map -> (B, D) unit fingerprint, a drop-in for gem_pooling.
    """
    from visual_pose.models.global_descriptor import AttentionPooling
    model = AttentionPooling(input_dim).to(DEVICE).eval()
    state = torch.load(REPO_DIR / "checkpoints" / run_name / checkpoint, map_location=DEVICE, weights_only=False)
    weights = state["model"]
    if any(k.startswith("pool.") for k in weights):   # trained inside SigmoidScaled: keep the pooling only
        weights = {k[len("pool."):]: v for k, v in weights.items() if k.startswith("pool.")}
    model.load_state_dict(weights)

    @torch.no_grad()
    def pool(backbone: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(model(backbone), dim=-1)
    return pool


class LearnedPooling:
    """
    load_learned_pooling deferred to the first call, so a module can name one
    (run.py's COMMON) without loading a checkpoint at import. Same contract:
    (B, 256, H', W') backbone map -> (B, D) unit fingerprint.
    """

    def __init__(self, run_name: str, checkpoint: str = "best.pt"):
        self.run_name, self.checkpoint, self._pool = run_name, checkpoint, None

    def __call__(self, backbone: torch.Tensor) -> torch.Tensor:
        if self._pool is None:
            self._pool = load_learned_pooling(self.run_name, checkpoint=self.checkpoint)
        return self._pool(backbone)

    def __repr__(self) -> str:
        return f"LearnedPooling({self.run_name!r})"
