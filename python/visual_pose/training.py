"""
The contrastive objective and the epoch loop.

Top of the stack: imports matching (for descriptors_at and the val metric) and
checkpoints (for saving). Nothing imports this except the entry point and sweeps.
"""
from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from visual_pose.checkpoints import save_checkpoint
from visual_pose.data_utils.constants import DEVICE, REPO_DIR
from visual_pose.data_utils.loaders import on_device
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


def train_val_model(model: nn.Module, train_data_loader: DataLoader,
                    val_data_loader: DataLoader, loss_fn: Callable,
                    optimizer: Optimizer, lr_scheduler: LRScheduler,
                    num_epochs: int, print_freq: int = 50,
                    temperature: float = 0.07,
                    checkpoint_dir: Path = REPO_DIR / "checkpoints",
                    checkpoint_tau: float = 8,
                    on_epoch_end: Callable[[int, float], None] | None = None,
                    eval_every: int = 1) -> nn.Module:
    """Train, evaluating on val every `eval_every` epochs. Returns the model.

        last.pt every epoch so a crash costs one epoch; best.pt only on
        improvement, so it survives later overfitting. on_epoch_end lets a
        sweep report progress or abandon a trial by raising.
        """
    model.to(DEVICE)
    best_mma = -1.0

    for epoch_i in range(num_epochs):
        # set the model in the train mode so the batch norm layers will behave correctly
        model.to(DEVICE)
        model.train()

        running_loss = 0.0
        for batch_i, batch_data in enumerate(on_device(train_data_loader, DEVICE)):
            images_i = batch_data["rgb_i"]   # (B, 3, H, W)
            images_j = batch_data["rgb_j"]
            uvs_i = batch_data["uv_i"]       # (B, N, 2) image pixels, (u, v)
            uvs_j = batch_data["uv_j"]       # (B, N, 2) the ground-t

            # matched descriptor pairs: row n of each is the SAME 3D point seen
            # from the two views. (B, N, D) each.
            sampled_i, _ = descriptors_at(model, images_i, uvs_i)
            sampled_j, _ = descriptors_at(model, images_j, uvs_j)

            loss = loss_fn(sampled_i, sampled_j, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # +1 so the first print averages a full window rather than one batch
            if (batch_i + 1) % print_freq == 0:
                last_lr = lr_scheduler.get_last_lr()[0]
                print(f'[epoch {epoch_i + 1}/{num_epochs}, batch {batch_i + 1:5d}/{len(train_data_loader)}] '
                      f'loss: {running_loss / print_freq:.3f} lr: {last_lr:.6f}')
                running_loss = 0.0
        lr_scheduler.step()

        # eval_every > 1 trades the per-epoch curve for wall clock; always
        # evaluate the last epoch so a run still ends with a real number
        if (epoch_i + 1) % eval_every != 0 and epoch_i != num_epochs - 1:
            continue

        val_errors = test_model(model, val_data_loader)
        val_mma = mma(val_errors, checkpoint_tau).item()

        # last.pt every epoch so a crash costs at most one epoch; best.pt only
        # when val actually improves, so it survives later overfitting.
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer,
                        lr_scheduler, epoch_i, val_mma)
        if val_mma > best_mma:
            best_mma = val_mma
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer,
                            lr_scheduler, epoch_i, val_mma)
            print(f"new best val MMA@{checkpoint_tau}: {val_mma:.2%} (epoch {epoch_i + 1})")

        # hook for a hyperparameter search to report progress and, if it wants,
        # abandon a trial early. Raising from the callback stops the run.
        if on_epoch_end is not None:
            on_epoch_end(epoch_i, val_mma)

    return model


def set_up_loss_optimizer_lr_scheduler(
        model: nn.Module, learning_rate: float, momentum: float, num_epochs: int,
        weight_decay: float = 1e-4, min_lr_factor: float = 0.01,
        optimizer: str = "sgd", temperature: float = 0.07,
) -> tuple[Callable, Optimizer, LRScheduler]:
    """Optimizer, cosine-annealed schedule, and the loss.

    Cosine rather than StepLR: StepLR's total decay is step_size times gamma,
    which desyncs from num_epochs and froze the last third of a run at lr/1000.
    """
    if optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=learning_rate,
                              momentum=momentum, weight_decay=weight_decay)
    elif optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                weight_decay=weight_decay)
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}, expected sgd|adamw")

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=num_epochs, eta_min=learning_rate * min_lr_factor)

    # partial rather than a lambda so it pickles, in case a loader ever carries it
    loss_fn = partial(infoNCE_loss, temperature=temperature)

    return loss_fn, opt, lr_scheduler
