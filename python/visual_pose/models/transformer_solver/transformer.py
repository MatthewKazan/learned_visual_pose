from pathlib import Path
from typing import Optional

import torch
from torch import nn, Tensor

from visual_pose.config.config import TransformerSolverConfig
from visual_pose.config.constants import DEVICE, REPO_DIR
from visual_pose.models.transformer_solver.tokenizer import Tokenizer
from visual_pose.models.transformer_solver.encoder import Encoder


class TransformerSolver(nn.Module):
    def __init__(
            self,
            descriptor_dim: int,
            num_heads: int,
            embed_dim: int,
            ff_dim: int,
            dropout: float,
            num_layers: int,
            use_descriptor: bool,
            pe_in_geometry: bool = False,
            closure_residual: bool = True,
            closure_edge_mlp: bool = True,
            zero_pose_features: bool = False,
            odometry_residual: bool = False,
            edge_to_vertex: bool = False,
            residual_scale: tuple[float, float] = (1.0, 1.0),
            descriptor_scale: float = 1.0,
            feature_stats: Optional[Path] = None,
    ):
        super(TransformerSolver, self).__init__()
        self.tokenizer = Tokenizer(embed_dim=embed_dim, descriptor_dim=descriptor_dim, use_descriptor=use_descriptor,
                                   pe_in_geometry=pe_in_geometry, closure_residual=closure_residual,
                                   closure_edge_mlp=closure_edge_mlp, zero_pose_features=zero_pose_features,
                                   odometry_residual=odometry_residual, edge_to_vertex=edge_to_vertex,
                                   residual_scale=residual_scale, descriptor_scale=descriptor_scale,
                                   feature_stats=feature_stats)
        self.encoder = Encoder(num_layers=num_layers, embed_dim=embed_dim, ff_dim=ff_dim, dropout=dropout, num_heads=num_heads)

        self.output = nn.Linear(embed_dim, 6) # Output corrective vector in tangent space of SE3 TODO: CONSIDER CLOSELY

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
        return self.output(pose_output) # (B, V, 6)

    @classmethod
    def from_config(cls, cfg: TransformerSolverConfig):
        model = cls(
            descriptor_dim=cfg.descriptor_dim,
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            ff_dim=cfg.ff_dim,
            dropout=cfg.dropout,
            num_layers=cfg.num_layers,
            use_descriptor=cfg.use_descriptor,
            pe_in_geometry=cfg.pe_in_geometry,
            closure_residual=cfg.closure_residual,
            closure_edge_mlp=cfg.closure_edge_mlp,
            zero_pose_features=cfg.zero_pose_features,
            odometry_residual=cfg.odometry_residual,
            edge_to_vertex=cfg.edge_to_vertex,
            residual_scale=(cfg.residual_scale_rho, cfg.residual_scale_phi),
            descriptor_scale=cfg.descriptor_scale,
            feature_stats=REPO_DIR / cfg.feature_stats if cfg.feature_stats else None,
        )
        path = cfg.starting_checkpoint_path
        if path is not None:
            if not path.exists():
                raise FileNotFoundError(f"starting_checkpoint {cfg.starting_checkpoint!r} -> {path} does not exist")
            state = torch.load(path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(state["model"])
            print(f"initialised from {path} (epoch {state.get('epoch', -1) + 1}, "
                  f"{state.get('metric_name', 'metric')} {state.get('val_metric', float('nan')):.4f})")
        return model











