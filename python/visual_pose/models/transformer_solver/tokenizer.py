import torch
from torch import nn, Tensor
from torch.nn import functional as F, MultiheadAttention
from typing import List, Any

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
    ):
        super(Tokenizer, self).__init__()
        self.from_edge = nn.Linear(embed_dim, embed_dim, bias=True)
        self.to_edge = nn.Linear(embed_dim, embed_dim, bias=True)

        self.embed_pose = nn.Linear(12, embed_dim, bias=True)
        self.embed_odometry = nn.Linear(33, embed_dim, bias=True)
        self.embed_closure = nn.Linear(33, embed_dim, bias=True)

        self.geometry_tokenizer = GeometryTokenizer(embed_dim=embed_dim, descriptor_dim=descriptor_dim, use_descriptor=use_descriptor)

        self.scene_embedding_token_init = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def project_all(self, poses, closures, odometry):
        pose_token = self.embed_pose(poses) # (B, V, 12) -> (B, V, F)
        odometry_token = self.embed_odometry(odometry) # (B, O, 33) -> (B, O, F)
        closure_token = self.embed_closure(closures) # (B, L, 33) -> (B, L, F)

        return pose_token, odometry_token, closure_token

    def forward(
            self,
            poses: Tensor,  # (B, V, 12) chained odometry
            points: Tensor,  # (B, V, P, 3)
            closures: Tensor,  # (B, L, 33)
            closures_ij: Tensor,  # (B, L, 2)
            odometry: Tensor, # (B, O, 33)
            odometry_ij: Tensor, # (B, O, 2)
            descriptor: Tensor, # (B, V, P, desc_dim)
    ) -> Tensor:
        # normalize so that rotation, translation and information are roughly the same scale
        closures = normalize_edges(closures)
        odometry = normalize_edges(odometry)
        poses = poses.clone()
        poses[..., 3:12:4] /= TRANSLATION_NORM
        pose_embedding, odometry_embedding, closure_embedding = self.project_all(poses=poses, closures=closures, odometry=odometry)
        # (V, embed_dim)
        pe = positional_encoding_1d(seq_len=pose_embedding.shape[1], input_dim=pose_embedding.shape[2], device=pose_embedding.device)
        geo_embedding = self.geometry_tokenizer(points, descriptor, pe)


        #          (B, V, embed_dim) + (B, V, embed_dim)
        sequence_tokens = pose_embedding + geo_embedding # (B, V, embed_dim)

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
    ):
        super(GeometryTokenizer, self).__init__()
        self.embed_dim = embed_dim
        self.use_descriptor = use_descriptor

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

        self.pe_scalar = nn.Parameter(torch.full((1,), 0.1))

        self.norm = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, points: Tensor, descriptor: Tensor, pe: Tensor) -> Tensor:
        """
        Encode points into one geometry vector per keyframe.

        :param points: (B, V, P, 3) points for each keyframe
        :param descriptor: (B, V, P, descriptor_dim) descriptor for each point
        :param pe: (V, F) positional encoding for each vertex

        :return: Token representing geometry of each keyframe (B, V, F)
        """
        B, V, P, _ = points.shape

        # Same as one Linear on the per-point token [x, y, z, d], split by columns:
        #   W [x; d] + b = [W_xyz  W_d] [x; d] + b = W_xyz x + W_d d + b
        # Split so each input gets its own init/normalisation, the descriptor
        # term can be skipped (geometry-only) or zero-initialised when added later.
        points_token = self.embed_points(points)  # (B, V, P, 3) -> (B, V, P, F)
        if self.use_descriptor:
            descriptor_token = self.embed_descriptor(descriptor)  # (B, V, P, desc_dim) -> (B, V, P, F)
            geo_token = points_token + descriptor_token  # (B, V, P, F) + (B, V, P, F) = (B, V, P, F)
        else:
            geo_token = points_token
        # normalize to stabilize scale differences between point and descriptor embeddings
        # nonlinear mlp needed to learn the interaction between point and descriptor embeddings
        geo_token = self.mlp(self.norm(geo_token))

        pe = pe * self.pe_scalar

        #           (B, V, P, F) + (1, V, 1, F)
        geo_token = geo_token + pe.unsqueeze(0).unsqueeze(2) # Every point gets the positional encoding of its vertex

        x = geo_token.flatten(0,1) # (B, V, P, F) -> (B*V, P, F) safe because it doesn't mix sequences and you can reseperate batches later
        q = self.w_query(self.query_vector.unsqueeze(0).unsqueeze(0)) # (1, 1, F) @ (F, F) -> (1, 1, F)

        k = self.w_key(x) # (B*V, P, F) @ (F, F)ᵀ -> (B*V, P, F)
        v = self.w_value(x) #(B*V, P, F) @ (F, F)ᵀ -> (B*V, P, F)

        #           softmax( [1, 1, F] @ [B*V, F, P] / (F ** .5) ) = (B*V, 1, P)
        alpha = torch.softmax(torch.matmul(q, k.transpose(1,2)) / (self.embed_dim ** .5), -1)
                # (B*V, 1, P) @ (B*V, P, F) -> (B*V, 1, F) -> (B*V, F) -> (B, V, F)
        pooled = torch.matmul(alpha, v).squeeze(1).reshape(points_token.shape[0], points_token.shape[1], -1)

        return self.output_layer(pooled) # return (B, V, F)







