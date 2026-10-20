import json
from pathlib import Path

import torch
from torch import nn, Tensor
from torch.nn import functional as F, MultiheadAttention
from typing import List, Any, Optional

from visual_pose.geometry.graph_math import relative, update_ground_truth_residual
from visual_pose.models.positional_encoding import positional_encoding_1d



_UPPER = torch.triu_indices(6, 6)   # edge_features' packing order: upper triangle, diagonal included
_STRICT_UPPER = torch.triu_indices(6, 6, offset=1)   # the 15 correlations; the diagonal's are all 1
# Clamp for the zero diagonal of padded edges: log(0) and 0/0 would be NaN, and NaN
# survives the attention mask (0 * NaN = NaN). Real diagonals are ~1e3 and up.
_MIN_INFORMATION = 1e-6
TRANSLATION_NORM = 10


def information_matrix(edges: Tensor) -> Tensor:
    """(..., 33) edge features -> (..., 6, 6) information matrix, from columns 12: (edge_features' upper triangle)."""
    upper = edges.new_zeros(*edges.shape[:-1], 6, 6)
    upper[..., _UPPER[0], _UPPER[1]] = edges[..., 12:]
    # mirror into the lower triangle; the diagonal is in both halves, so remove one copy
    return upper + upper.transpose(-2, -1) - torch.diag_embed(torch.diagonal(upper, dim1=-2, dim2=-1))


def normalize_edges(edges: Tensor) -> Tensor:
    """
    (..., 33) -> (..., 33): columns 12: become log of the information diagonal (6)
    then the correlations above it (15), C_ij = I_ij / sqrt(I_ii I_jj) in [-1, 1].
    Invertible, so nothing is lost; a new tensor, the input is left untouched.
    """
    info = information_matrix(edges)  # (B, E, 6, 6)
    d = torch.diagonal(info, dim1=-2, dim2=-1).clamp(min=_MIN_INFORMATION)  # (B, E, 6)
    log = torch.log(d)
    s = torch.sqrt(d)
    denom = s[..., :, None] * s[..., None, :]  # entry (i,j) = s_i * s_j
    C = info / denom
    norm_info = torch.cat((log, C[..., _STRICT_UPPER[0], _STRICT_UPPER[1]]), dim=-1)  # (B, E, 6 + 15)
    edges = torch.cat((edges[..., :12], norm_info), dim=-1)
    edges[..., 3:12:4] /= TRANSLATION_NORM
    return edges


class Tokenizer(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            descriptor_dim: int,
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
        super(Tokenizer, self).__init__()
        self.pe_in_geometry = pe_in_geometry
        self.closure_residual = closure_residual
        self.closure_edge_mlp = closure_edge_mlp
        self.zero_pose_features = zero_pose_features
        self.odometry_residual = odometry_residual
        self.edge_to_vertex = edge_to_vertex
        self.use_descriptor = use_descriptor

        if edge_to_vertex:
            # graph-network node update: every edge sends a message to each endpoint, a vertex adds the
            # mean of what it receives. Input [edge, this end, other end], so start and end differ.
            self.vertex_msg_mlp = nn.Sequential(
                nn.Linear(3 * embed_dim, 2 * embed_dim),
                nn.GELU(),
                nn.Linear(2 * embed_dim, embed_dim),
            )
            with torch.no_grad():   # starts as no message: training begins from the model without it
                self.vertex_msg_mlp[-1].weight.zero_()
                self.vertex_msg_mlp[-1].bias.zero_()
        self.from_edge = nn.Linear(embed_dim, embed_dim, bias=True)
        self.to_edge = nn.Linear(embed_dim, embed_dim, bias=True)

        self.embed_pose = nn.Linear(12, embed_dim, bias=True)
        self.embed_odometry = nn.Linear(33 + 6 if odometry_residual else 33, embed_dim, bias=True)
        self.embed_closure = nn.Linear(33 + 6 if closure_residual else 33, embed_dim, bias=True)
        if closure_edge_mlp:
            # e' = e + MLP([e, v_i, v_j]): each closure with its two endpoint vertex tokens, gathered by
            # index rather than found through attention over positional codes
            self.edge_mlp = nn.Sequential(
                nn.Linear(3 * embed_dim, 2 * embed_dim),
                nn.GELU(),
                nn.Linear(2 * embed_dim, embed_dim),
            )
            with torch.no_grad():   # starts as e' = e: training begins from the model without it
                self.edge_mlp[-1].weight.zero_()
                self.edge_mlp[-1].bias.zero_()



        self.scene_embedding_token_init = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.feature_stats = None

        if feature_stats is not None:
            # stats were measured with the hand scales at 1; applying both would scale twice
            self.register_buffer("residual_scale", torch.ones(6), persistent=False)
            self.descriptor_scale = 1.0
            self.feature_stats = feature_stats
            with open(feature_stats) as f:
                d = json.load(f)
            for k in ['pose', 'descriptors', 'odometry', 'closure', 'points']:
                self.register_buffer(f"{k}_mean", torch.tensor(d[k]['mean'], dtype=torch.float32))
                # floor: a constant column would divide by 0. Smallest measured std 0.003 (odometry residual rho)
                self.register_buffer(f"{k}_std", torch.tensor(d[k]['std'], dtype=torch.float32).clamp(min=1e-3))
        else:
            # (rho, phi) multipliers on every residual: TransformerSolverConfig.residual_scale_*
            self.register_buffer("residual_scale", torch.tensor([residual_scale[0]] * 3 + [residual_scale[1]] * 3),
                                 persistent=False)
            self.descriptor_scale = descriptor_scale

        self.geometry_tokenizer = GeometryTokenizer(embed_dim=embed_dim, descriptor_dim=descriptor_dim, use_descriptor=use_descriptor,
                                                    pe_in_geometry=pe_in_geometry, descriptor_scale=self.descriptor_scale)



    def project_all(self, poses, closures, odometry):
        pose_token = self.embed_pose(poses) # (B, V, 12) -> (B, V, F)
        odometry_token = self.embed_odometry(odometry) # (B, O, 33) -> (B, O, F)
        closure_token = self.embed_closure(closures) # (B, L, 33 [+ 6 residual]) -> (B, L, F)

        return pose_token, odometry_token, closure_token

    def get_edge_residual(
            self,
            poses,
            closures,
            closures_ij
    ):
        poses_T = poses.unflatten(-1, (3,4))
        poses_R, pose_t = poses_T[..., :3, :3], poses_T[..., :3, 3]
        pose_relative_R, pose_relative_t = relative(poses_R, pose_t, closures_ij)
        pose_relative_T = torch.cat([pose_relative_R, pose_relative_t.unsqueeze(-1)], dim=-1)

        _, residual_log = update_ground_truth_residual(closures[..., :12], pose_relative_T)


        return residual_log * self.residual_scale

    def scale_features(
            self,
            poses: Tensor,  # (B, V, 12) chained odometry
            descriptor: Tensor,  # (B, V, P, desc_dim)
            odometry: Tensor,  # (B, O, 33 or 39)
            closures: Tensor,  # (B, L, 33 or 39)
            points: Tensor,  # (B, V, P, 3)
    ):
        poses =(poses - self.pose_mean) / self.pose_std

        odom_mean = self.odometry_mean
        odom_std = self.odometry_std
        if not self.odometry_residual:
            odom_mean = self.odometry_mean[..., :33]
            odom_std = self.odometry_std[..., :33]

        closure_mean = self.closure_mean
        closure_std = self.closure_std
        if not self.closure_residual:
            closure_mean = self.closure_mean[..., :33]
            closure_std = self.closure_std[..., :33]

        odometry = (odometry - odom_mean) / odom_std
        closures = (closures - closure_mean) / closure_std
        points = (points - self.points_mean) / self.points_std

        if self.use_descriptor:
            descriptor = (descriptor - self.descriptors_mean) / self.descriptors_std

        return poses, descriptor, odometry, closures, points

    def forward(
            self,
            poses: Tensor,  # (B, V, 12) chained odometry
            points: Tensor,  # (B, V, P, 3)
            closures: Tensor,  # (B, L, 33)
            closures_ij: Tensor,  # (B, L, 2)
            odometry: Tensor, # (B, O, 33)
            odometry_ij: Tensor, # (B, O, 2)
            descriptor: Tensor, # (B, V, P, desc_dim)
            odom_mask: Optional[Tensor] = None,  # (B, O) bool, True = padding; needed by edge_to_vertex
            closure_mask: Optional[Tensor] = None,  # (B, L)
    ) -> Tensor:
        # normalize so that rotation, translation and information are roughly the same scale
        closures = normalize_edges(closures)
        odometry = normalize_edges(odometry)
        poses = poses.clone()
        poses[..., 3:12:4] /= TRANSLATION_NORM
        if self.closure_residual:   # after the scaling above: Z and the poses in the same units
            closures = torch.cat((closures, self.get_edge_residual(poses, closures, closures_ij)), dim=-1)
        if self.odometry_residual:   # exactly 0 on chained odometry; non-zero once the poses are perturbed or refined
            odometry = torch.cat((odometry, self.get_edge_residual(poses, odometry, odometry_ij)), dim=-1)
        if self.feature_stats:
            poses, descriptor, odometry, closures, points = self.scale_features(poses, descriptor, odometry, closures, points)
        # zero_pose_features: the vertex token gets no absolute pose; the residual above still uses it
        pose_embedding, odometry_embedding, closure_embedding = self.project_all(
            poses=torch.zeros_like(poses) if self.zero_pose_features else poses, closures=closures, odometry=odometry)
        # (V, embed_dim)
        pe = positional_encoding_1d(seq_len=pose_embedding.shape[1], input_dim=pose_embedding.shape[2], device=pose_embedding.device)
        geo_embedding = self.geometry_tokenizer(points, descriptor, pe if self.pe_in_geometry else None)

        #          (B, V, embed_dim) + (B, V, embed_dim) + (1, V, embed_dim)
        sequence_tokens = pose_embedding + geo_embedding
        if not self.pe_in_geometry:
            sequence_tokens = sequence_tokens + pe.unsqueeze(0)

        if self.closure_edge_mlp:
            b = torch.arange(closures_ij.shape[0], device=closures_ij.device)[:, None]   # (B, 1)
            v_i, v_j = sequence_tokens[b, closures_ij[..., 0]], sequence_tokens[b, closures_ij[..., 1]]   # (B, L, F) each
            closure_embedding = closure_embedding + self.edge_mlp(torch.cat((closure_embedding, v_i, v_j), dim=-1))

        if self.edge_to_vertex:
            e = torch.cat((odometry_embedding, closure_embedding), dim=1)            # (B, E, F)
            ij = torch.cat((odometry_ij, closures_ij), dim=1)                        # (B, E, 2)
            real = (~torch.cat((odom_mask, closure_mask), dim=1)).unsqueeze(-1).to(e.dtype)   # (B, E, 1): padding sends nothing
            b = torch.arange(ij.shape[0], device=ij.device)[:, None]
            v_i, v_j = sequence_tokens[b, ij[..., 0]], sequence_tokens[b, ij[..., 1]]
            received = torch.zeros_like(sequence_tokens)                              # (B, V, F)
            degree = sequence_tokens.new_zeros(sequence_tokens.shape[:2] + (1,))      # (B, V, 1)
            for end, here, there in ((ij[..., 0], v_i, v_j), (ij[..., 1], v_j, v_i)):
                msg = self.vertex_msg_mlp(torch.cat((e, here, there), dim=-1)) * real
                received = received.scatter_add(1, end.unsqueeze(-1).expand_as(msg), msg)
                degree = degree.scatter_add(1, end.unsqueeze(-1), real)
            sequence_tokens = sequence_tokens + received / degree.clamp(min=1)

        # Combine odometry and closure embeddings
        edge_embedding = torch.cat((odometry_embedding, closure_embedding), dim=1) # (B, E, embed_dim)

        edge_ij = torch.cat([odometry_ij, closures_ij], dim=1)
        starts = self.from_edge(pe[edge_ij[:, :, 0]])
        ends = self.to_edge(pe[edge_ij[:, :, 1]])

        edge_token = edge_embedding + starts + ends

        scene_token = self.scene_embedding_token_init.expand(pose_embedding.shape[0], -1, -1) # (B, 1, embed_dim)

        return torch.concat([sequence_tokens, edge_token, scene_token], dim=1) # (B, V+E+1, embed_dim)


class GeometryTokenizer(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            descriptor_dim: int,
            use_descriptor: bool,
            pe_in_geometry: bool = False,
            descriptor_scale: float = 1.0,
    ):
        super(GeometryTokenizer, self).__init__()
        self.embed_dim = embed_dim
        self.use_descriptor = use_descriptor
        self.descriptor_scale = descriptor_scale   # TransformerSolverConfig.descriptor_scale
        if pe_in_geometry:   # old tokenizer only: its checkpoints carry this parameter
            self.pe_scalar = nn.Parameter(torch.full((1,), 0.1))

        self.embed_points = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.embed_descriptor = nn.Linear(descriptor_dim, embed_dim, bias=True)

        self.query_vector = nn.Parameter(torch.randn(embed_dim))  # (F), pooling query, asks, what keys are most relevant to pooling

        self.w_key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.output_layer = nn.Linear(embed_dim, embed_dim, bias=True)

        self.norm = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, points: Tensor, descriptor: Tensor, pe: Optional[Tensor] = None) -> Tensor:
        """
        Encode points into one geometry vector per keyframe.

        :param points: (B, V, P, 3) points for each keyframe
        :param descriptor: (B, V, P, descriptor_dim) descriptor for each point

        :return: Token representing geometry of each keyframe (B, V, F)
        """
        B, V, P, _ = points.shape

        # Same as one Linear on the per-point token [x, y, z, d], split by columns:
        #   W [x; d] + b = [W_xyz  W_d] [x; d] + b = W_xyz x + W_d d + b
        # Split so each input gets its own init/normalisation, the descriptor
        # term can be skipped (geometry-only) or zero-initialised when added later.
        points_token = self.embed_points(points)  # (B, V, P, 3) -> (B, V, P, F)
        if self.use_descriptor:
            descriptor_token = self.embed_descriptor(descriptor * self.descriptor_scale)  # (B, V, P, desc_dim) -> (B, V, P, F)
            geo_token = points_token + descriptor_token  # (B, V, P, F) + (B, V, P, F) = (B, V, P, F)
        else:
            geo_token = points_token
        # normalize to stabilize scale differences between point and descriptor embeddings
        # nonlinear mlp needed to learn the interaction between point and descriptor embeddings
        geo_token = self.mlp(self.norm(geo_token))
        if pe is not None:   # old tokenizer: every point of vertex i gets pe_i
            #           (B, V, P, F) + (1, V, 1, F)
            geo_token = geo_token + (pe * self.pe_scalar).unsqueeze(0).unsqueeze(2)

        x = geo_token.flatten(0,1) # (B, V, P, F) -> (B*V, P, F) safe because it doesn't mix sequences and you can reseperate batches later
        q = self.w_query(self.query_vector.unsqueeze(0).unsqueeze(0)) # (1, 1, F) @ (F, F) -> (1, 1, F)

        k = self.w_key(x) # (B*V, P, F) @ (F, F)ᵀ -> (B*V, P, F)
        v = self.w_value(x) #(B*V, P, F) @ (F, F)ᵀ -> (B*V, P, F)

        #           softmax( [1, 1, F] @ [B*V, F, P] / (F ** .5) ) = (B*V, 1, P)
        alpha = torch.softmax(torch.matmul(q, k.transpose(1,2)) / (self.embed_dim ** .5), -1)
                # (B*V, 1, P) @ (B*V, P, F) -> (B*V, 1, F) -> (B*V, F) -> (B, V, F)
        pooled = torch.matmul(alpha, v).squeeze(1).reshape(points_token.shape[0], points_token.shape[1], -1)

        return self.output_layer(pooled) # return (B, V, F)







