"""
The descriptor CNN's pieces for training.fit: its contrastive objective, one
batch's loss, and the checkpoint metric.
"""
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader

from visual_pose.matching import descriptors_at, mma, test_model


def infoNCE_loss(sampled_i: Tensor, sampled_j: Tensor,
                 temperature: float = 0.07) -> Tensor:
    """InfoNCE over matched pairs, (B, N, D) each, row n the same 3D point.

        L_n = -log[ exp(S[n,n]/t) / sum_k exp(S[n,k]/t) ]

    That fraction is a softmax over row n, so cross_entropy(S/t, labels) is the
    same thing with log-sum-exp stabilisation. Untrained loss ~= log(N).
    """
    B, N, _ = sampled_i.shape

    # unit-length descriptors, so the dot product IS the cosine. (B, N, N)
    similarity = torch.matmul(sampled_i, sampled_j.transpose(-1, -2))

    # cosines span only [-1,1]; softmax of that is near-uniform and the
    # gradient can't separate positive from negatives. /t stretches to ~[-14,14]
    logits = similarity / temperature

    # row n's positive is column n -- the diagonal -- so the label for row n IS
    # n, hence arange. Flattened row (b*N + n) is query n of pair b, so labels
    # tile across b: repeat, not repeat_interleave.
    labels = torch.arange(N, device=similarity.device).repeat(B)

    # flatten to 2D: cross_entropy's K-dim form wants classes on axis 1, ours
    # are on axis 2, and both are N so getting it wrong would run silently.
    # Second term asks the reverse question (argmax isn't symmetric).
    loss_i_to_j = F.cross_entropy(logits.flatten(0, 1), labels)
    loss_j_to_i = F.cross_entropy(logits.transpose(-1, -2).flatten(0, 1), labels)
    return 0.5 * (loss_i_to_j + loss_j_to_i)


def cnn_step(model: nn.Module, batch: dict[str, Tensor], temperature: float) -> Tensor:
    """One TACorrespondenceDataset batch -> InfoNCE loss."""
    # matched descriptor pairs: row n of each is the SAME 3D point seen
    # from the two views. (B, N, D) each.
    sampled_i, _ = descriptors_at(model, batch["rgb_i"], batch["uv_i"])
    sampled_j, _ = descriptors_at(model, batch["rgb_j"], batch["uv_j"])
    return infoNCE_loss(sampled_i, sampled_j, temperature=temperature)


def cnn_evaluate(model: nn.Module, val_loader: DataLoader, tau: float) -> float:
    """MMA@tau over val_loader."""
    return mma(test_model(model, val_loader), tau).item()
