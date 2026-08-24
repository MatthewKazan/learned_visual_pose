from functools import partial

import numpy as np
import torch
from torch import floor, Tensor
import torch.nn.functional as F
from visual_pose.data_utils.constants import DEVICE, REPO_DIR



# Dimension glossary used throughout:
#   B  batch size (frame pairs)      N  correspondences per pair (512)
#   D  descriptor dim (128)          M  candidates per image = H'*W' (4800)
#   H, W  image size (480, 640)      H', W'  feature-map size (60, 80)
# Primed names (u', v') are feature-map coordinates; unprimed are image pixels.


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


def mma(error: Tensor, tau):
    return torch.mean((error < tau).float())


def save_checkpoint(path, model, optimizer, lr_scheduler, epoch, val_mma):
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


def bilinear_sample(descriptor: Tensor, feature_uvs: Tensor):
    """descriptor (B,D,H',W'), feature_uvs (B,N,2) in feature coords -> (B,N,D)"""
    u0 = floor(feature_uvs[...,0])   # (B, N)
    v0 = floor(feature_uvs[...,1])   # (B, N)

    # change from (B, C, H, W) to (B, H, W, C) so a descriptor is a trailing
    # slice -- lets the gather below return (B, N, D) directly
    descriptor = descriptor.permute(0, 2, 3, 1)

    # fractional parts, unsqueezed to (B, N, 1) so they broadcast across D
    a = (feature_uvs[..., 0] - u0).unsqueeze(-1)
    b = (feature_uvs[..., 1] - v0).unsqueeze(-1)

    u1 = (u0 + 1).clamp(0, descriptor.shape[2] - 1)
    v1 = (v0 + 1).clamp(0, descriptor.shape[1] - 1)
    u0 = u0.clamp(0, descriptor.shape[2] - 1)
    v0 = v0.clamp(0, descriptor.shape[1] - 1)

    # (B, 1) so it broadcasts against the (B, N) coord tensors. Without it the
    # gather returns the cross product (B, D, B, N) instead of the diagonal.
    batch = torch.arange(descriptor.shape[0]).unsqueeze(-1).to(descriptor.device)

    # each term: (B,N,1) weight * (B,N,D) corner -> (B, N, D)
    w1 = (1 - a) * (1 - b) * descriptor[batch, v0.long(), u0.long()]
    w2 = (1 - a) * b *       descriptor[batch, v1.long(), u0.long()]
    w3 = a * (1 - b) *       descriptor[batch, v0.long(), u1.long()]
    w4 = a * b *             descriptor[batch, v1.long(), u1.long()]

    # blending unit vectors gives sub-unit length, so re-normalize over D
    return F.normalize(w1 + w2 + w3 + w4, dim=-1)


def descriptors_at(model, images, uvs):
    """
    Run the model and read descriptors at the given image pixels.

    images (B,3,H,W), uvs (B,N,2) in *image* pixels, (u, v) order.
    Returns (sampled (B,N,D), feature_map (B,D,H',W')).

    The feature map comes back too because the eval path needs the whole map as
    a candidate pool, while the loss path only needs the sampled descriptors.
    """
    descriptors = model(images)

    # image pixels -> feature coords: u scales by width, v by height.
    # Getting these two crossed is silent whenever the strides are uniform,
    # which is why it lives in one place.
    feature_uvs = torch.stack([
        uvs[..., 0] * (descriptors.shape[3] / images.shape[3]),
        uvs[..., 1] * (descriptors.shape[2] / images.shape[2]),
    ], dim=-1)

    return bilinear_sample(descriptors, feature_uvs), descriptors


def descriptor_error(descriptors_i, descriptors_j, images_j, uvs_j):
    # the answer pool: image j's whole map as a flat list of candidates.
    # (B,D,H',W') -> (B,H',W',D) -> (B, M, D) with M = H'*W'.
    # Row-major collapse puts W' on the fast axis, hence the div/mod below.
    candidate_j = descriptors_j.permute(0, 2, 3, 1).flatten(1, 2)

    # (B,N,D) @ (B,D,M) -> (B, N, M). Descriptors are unit length, so the
    # dot product IS the cosine. argmax over M -> (B, N) winning indices.
    similarity = torch.matmul(descriptors_i, candidate_j.transpose(-1, -2))
    best_match = torch.argmax(similarity, dim=-1)

    # unpack the flat index into a feature cell: (B, N) each
    W_feat = descriptors_j.shape[3]
    v_prime = best_match // W_feat
    u_prime = best_match % W_feat

    # feature cell -> image pixel, undoing the scaling applied above.
    # (B, N) each; still needs stacking to (B, N, 2) as (u, v) to match uvs_j.
    v_pixel = v_prime * (images_j.shape[2] / descriptors_j.shape[2])
    u_pixel = u_prime * (images_j.shape[3] / descriptors_j.shape[3])

    # (B, N, 2) as (u, v) to match uvs_j's layout
    predicted_uvs = torch.stack([u_pixel, v_pixel], dim=-1)
    # signed, (B, N, 2): mean should sit near 0 in both u and v. A
    # consistent +-4 means a half-cell offset in the coord convention,
    # which MMA alone would hide as "mediocre".
    offset = predicted_uvs - uvs_j
    # (B, N) Euclidean pixel distance per correspondence
    pixel_error = torch.linalg.vector_norm(offset, dim=-1)
    return pixel_error, offset



def test_model(model, data_loader, taus=(1, 3, 4, 8, 12, 16), identity=False):
    """
    Compute matching accuracy (MMA) of the model.

    Inputs:
      - model: A CNN implemented in PyTorch
      - data_loader: A data loader that will provide batched images and labels
      - taus: pixel thresholds to report MMA at
      - identity: if True, match each image against ITSELF instead of its pair.
                  The correct answer is then "the nearest grid cell to the
                  query", so MMA should hit the quantization ceiling. Any
                  shortfall is an indexing/convention bug, not a weak
                  descriptor -- which a normal run can't distinguish.
    """

    # .to() on a Module mutates in place -- no reassignment needed
    model.to(DEVICE)
    # set the model in evaluation mode so the batch norm layers will behave correctly
    model.eval()

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        errors, offsets = [], []
        stride = None
        for batch_data in on_device(data_loader, DEVICE):

            images_i = batch_data["rgb_i"]   # (B, 3, H, W)
            images_j = batch_data["rgb_j"]
            uvs_i = batch_data["uv_i"]       # (B, N, 2) image pixels, (u, v)
            uvs_j = batch_data["uv_j"]       # (B, N, 2) the ground-truth answers

            if identity:
                images_j, uvs_j = images_i, uvs_i

            # the queries: one descriptor per ground-truth correspondence
            sampled_descriptors_i, descriptors_i = descriptors_at(model, images_i, uvs_i)
            descriptors_j = model(images_j)   # (B, D, H', W')

            pixel_error, offset = descriptor_error(sampled_descriptors_i, descriptors_j, images_j, uvs_j)
            errors.append(pixel_error.flatten())
            offsets.append(offset.flatten(0, 1))
            stride = images_j.shape[3] / descriptors_j.shape[3]

        errors = torch.cat(errors)             # (total_correspondences,)
        offsets = torch.cat(offsets)           # (total_correspondences, 2)
        H, W = images_i.shape[2], images_i.shape[3]

        # Only inliers say anything about the coordinate convention -- the
        # outlier tail dominates the raw mean and buries a real +-4 bias.
        inlier = errors < stride
        bias = offsets[inlier].mean(dim=0)
        print(f"\n{'identity check' if identity else 'pair matching'}"
              f"  |  {len(errors)} correspondences, stride {stride:.0f}")
        print(f"mean signed offset over inliers (err < {stride:.0f} px, n={inlier.sum()}): "
              f"u {bias[0]:+.2f}  v {bias[1]:+.2f} px"
              f"   (near 0 = coordinate convention is consistent)")
        print(f"median error: {errors.median():.1f} px")
        print(f"\n{'tau':>5} {'MMA':>8} {'random':>9} {'ceiling':>8}")
        for tau in taus:
            # a random cell lands within tau with prob ~ pi*tau^2 / (H*W);
            # the ceiling is the fraction of one stride-by-stride cell that
            # sits within tau of its own center.
            floor_ = np.pi * tau ** 2 / (H * W)
            ceiling = min(1.0, np.pi * tau ** 2 / stride ** 2)
            print(f"{tau:>5} {mma(errors, tau):>8.2%} {floor_:>9.3%} {ceiling:>8.1%}")
        return errors


def infoNCE_loss(sampled_i, sampled_j, temperature=0.07):
    """
    InfoNCE over matched descriptor pairs, (B, N, D) each, where row n of both
    is the same 3D point from two views.

        L_n = -log[ exp(S[n,n]/t) / sum_k exp(S[n,k]/t) ]

    That fraction is a softmax over row n, and -log of the target's entry is
    cross_entropy -- so cross_entropy(S/t, labels) IS the formula above, with
    log-sum-exp stabilization for free.

    Sanity: loss ~= log(N) (6.24 at N=512) untrained, ~0 when perfect.
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


def train_val_model(model, train_data_loader, val_data_loader, loss_fn, optimizer,
                    lr_scheduler, num_epochs, print_freq=50, temperature=0.07,
                    checkpoint_dir=REPO_DIR / "checkpoints", checkpoint_tau=8,
                    on_epoch_end=None, eval_every=1):
    """
    Training and validating a CNN model using PyTorch.

    Inputs:
      - model: A CNN implemented in PyTorch
      - data_loader: A data loader that will provide batched images and labels
      - loss_fn: A loss function (e.g., cross entropy loss)
      - lr_scheduler: Learning rate scheduler
      - num_epochs: Number of epochs in total
      - print_freq: Frequency to print training statistics
      - checkpoint_dir: where last.pt / best.pt go
      - checkpoint_tau: the MMA threshold "best" is judged on

    Output:
      - model: Trained CNN model
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


def set_up_loss_optimizer_lr_scheduler(model, learning_rate, momentum, num_epochs,
                                       weight_decay=1e-4, min_lr_factor=0.01,
                                       optimizer="sgd", temperature=0.07):
    """
    Optimizer + cosine-annealed learning rate + the loss, ready to hand to
    train_val_model.

    Cosine rather than StepLR because StepLR's total decay is step_size and gamma
    multiplied out, which desyncs from num_epochs silently -- step_size=n/3 with
    gamma=0.1 always lands at lr/1000 and spends the last third of the run frozen.
    Cosine takes T_max=num_epochs and decays smoothly to eta_min, so the schedule
    always spans exactly the run.

    weight_decay is plain L2 regularization, aimed at the train/val gap.
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

