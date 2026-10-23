from typing import Optional

import torch
from torch import nn, Tensor

from visual_pose.config.config import TransformerSolverConfig
from visual_pose.config.constants import DEVICE
from visual_pose.models.transformer_solver.tokenizer import Tokenizer
from visual_pose.models.transformer_solver.encoder import Encoder


class TransformerSolver(nn.Module):
    def __init__(self, cfg: TransformerSolverConfig):
        super(TransformerSolver, self).__init__()
        self.cfg = cfg   # plain attribute, not in the state_dict
        self.tokenizer = Tokenizer(cfg)
        self.encoder = Encoder(num_layers=cfg.num_layers, embed_dim=cfg.embed_dim, ff_dim=cfg.ff_dim,
                               dropout=cfg.dropout, num_heads=cfg.num_heads)

        self.output = nn.Linear(cfg.embed_dim, 6) # Output corrective vector in tangent space of SE3 TODO: CONSIDER CLOSELY

        with torch.no_grad():
            self.output.weight.zero_()
            self.output.bias.zero_()


    def forward(self,
                poses: Tensor,  # (B, V, 12) chained odometry
                points: Tensor,  # (B, V, P, 3)
                closures: Tensor,  # (B, L, 33)  closures
                closures_ij: Tensor,  # (B, L, 2)
                odometry: Tensor, # (B, O, 33)
                odometry_ij: Tensor, # (B, O, 2)
                descriptors: Optional[Tensor], # (B, V, P, desc_dim) fp16 as stored, or None when off
                vertex_mask: Tensor,  # (B, V) bool, True = padding
                closure_mask: Tensor,  # (B, L) bool, True = padding
                odom_mask: Tensor,  # (B, O) bool, True = padding

    ) -> Tensor:  # (B, V, 6) correction vector in tangent space of SE(3), apply as right perturb to current best guess
        if descriptors is not None:
            descriptors = descriptors.float()
        embeddings = self.tokenizer(poses, points, closures, closures_ij, odometry, odometry_ij, descriptors,
                                    odom_mask=odom_mask, closure_mask=closure_mask) # (B, V+E+1, embed_dim)
        scene_mask = torch.zeros(embeddings.shape[0], 1, dtype=torch.bool, device=embeddings.device)  # never padding

        mask = torch.cat([vertex_mask, odom_mask, closure_mask, scene_mask], dim=1)

        x = self.encoder(embeddings, mask)# (B, V+E+1, embed_dim)
        pose_output = x[:, :poses.shape[1], :] # (B, V, embed_dim)
        delta = self.output(pose_output) # (B, V, 6)
        delta = torch.cat(
            [torch.zeros_like(delta[:, :1]), delta[:, 1:]], # Fix vertex 0
            dim=1
        )
        return delta

    @classmethod
    def from_config(cls, cfg: TransformerSolverConfig):
        model = cls(cfg)
        path = cfg.starting_checkpoint_path
        if path is not None:
            if not path.exists():
                raise FileNotFoundError(f"starting_checkpoint {cfg.starting_checkpoint!r} -> {path} does not exist")
            state = torch.load(path, map_location=DEVICE, weights_only=False)
            # a starting checkpoint may predate whitened_residual: its layer starts from __init__ (zero-init),
            # nothing else may be missing or extra
            missing, unexpected = model.load_state_dict(state["model"], strict=False)
            if unexpected:
                raise RuntimeError(f"{path} does not match the model:  unexpected {unexpected}")
            if missing:
                print(f"WARNING: missing {missing}")
            print(f"initialised from {path} (epoch {state.get('epoch', -1) + 1}, "
                  f"{state.get('metric_name', 'metric')} {state.get('val_metric', float('nan')):.4f})")
        return model











