"""
Checkpoint round-trip: saved by training, loaded by every evaluation script.
"""
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from visual_pose.config.config import Config
from visual_pose.config.constants import DEVICE, REPO_DIR
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


def load_transformer(run_name: str, checkpoint: str = "best.pt"):
    """
    (model in eval mode on DEVICE, its TransformerSolverConfig) from checkpoints/<run_name>/: the
    config.json the run saved, so architecture flags (pe_in_geometry, rotate_points, ...) match the
    weights. Fields a newer config added and the run predates keep their defaults.
    """
    import dataclasses, json
    from visual_pose.config.config import TransformerSolverConfig
    from visual_pose.models.transformer_solver.transformer import TransformerSolver
    run_dir = REPO_DIR / "checkpoints" / run_name
    known = {f.name for f in dataclasses.fields(TransformerSolverConfig)}
    saved = json.loads((run_dir / "config.json").read_text())
    cfg = TransformerSolverConfig(**{k: v for k, v in saved.items() if k in known})
    cfg = dataclasses.replace(cfg, starting_checkpoint=None)   # from_config must not load a different checkpoint first
    state = torch.load(run_dir / checkpoint, map_location=DEVICE, weights_only=False)["model"]
    if "pe_in_geometry" not in saved:
        # run predates the flags (2026-10-16): only the old tokenizer has pe_scalar, and only the new
        # one rotated its points (rebuilt_subset_e16); defaulting would build the wrong architecture
        old = "tokenizer.geometry_tokenizer.pe_scalar" in state
        cfg = dataclasses.replace(cfg, pe_in_geometry=old, rotate_points=not old)
    if "closure_residual" not in saved:   # predates 2026-10-19: embed_closure is 33 + 6 wide only with it
        cfg = dataclasses.replace(cfg, closure_residual=state["tokenizer.embed_closure.weight"].shape[1] == 39)
    if "closure_edge_mlp" not in saved:   # predates 2026-10-19: no endpoint-gathering closure MLP
        cfg = dataclasses.replace(cfg, closure_edge_mlp="tokenizer.edge_mlp.0.weight" in state)
    if "zero_pose_features" not in saved:   # every run before the flag fed the poses
        cfg = dataclasses.replace(cfg, zero_pose_features=False)
    if "odometry_residual" not in saved:
        cfg = dataclasses.replace(cfg, odometry_residual=state["tokenizer.embed_odometry.weight"].shape[1] == 39)
    if "edge_to_vertex" not in saved:   # edge -> vertex messages
        cfg = dataclasses.replace(cfg, edge_to_vertex=False)
    if "residual_scale_rho" not in saved:   # unscaled before 2026-10-19 (not a weight: cannot be inferred)
        cfg = dataclasses.replace(cfg, residual_scale_rho=1.0, residual_scale_phi=1.0)
    if "descriptor_scale" not in saved:
        cfg = dataclasses.replace(cfg, descriptor_scale=1.0)
    if "pose_noise_deg" not in saved:
        cfg = dataclasses.replace(cfg, pose_noise_deg=0.0, pose_noise_m=0.0)
    if "feature_stats" not in saved:   # predates 2026-10-20: standardized runs carry the stats as buffers
        cfg = dataclasses.replace(cfg, feature_stats="out/feature_stats.json" if "tokenizer.pose_mean" in state else None)
    if "whitened_residual" not in saved:   # predates 2026-10-22
        cfg = dataclasses.replace(cfg, whitened_residual="tokenizer.embed_whitened_residual.weight" in state)
    model = TransformerSolver.from_config(cfg).to(DEVICE)
    model.load_state_dict(state)
    return model.eval(), cfg


def load_learned_pooling(run_name: str = "attention_pooling_infonce", input_dim: int = 256,
                         checkpoint: str = "best.pt"):
    """
    A trained AttentionPooling as a WorldModel pooling_method: (B, 256, H', W')
    backbone map -> (B, D) unit fingerprint, a drop-in for gem_pooling.
    """
    from visual_pose.models.global_descriptor import AttentionPooling2d
    model = AttentionPooling2d(input_dim).to(DEVICE).eval()
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
