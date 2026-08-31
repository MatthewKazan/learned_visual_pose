"""
Reading descriptors out of a feature map, and scoring how well they match.

Sits below both training and evaluation: the loss path calls descriptors_at to
get matched pairs, the eval path calls it to get queries. Nothing here imports a
model class -- test_model takes whatever it is handed, which is what lets SIFT
go through the same metric as the CNN.
"""
import numpy as np
import torch
from torch import floor, Tensor
import torch.nn.functional as F

from visual_pose.data_utils.constants import DEVICE
from visual_pose.data_utils.loaders import on_device
from visual_pose.geometry.best_match import best_match


# Dimension glossary used throughout:
#   B  batch size (frame pairs)      N  correspondences per pair (512)
#   D  descriptor dim (128)          M  candidates per image = H'*W' (4800)
#   H, W  image size (480, 640)      H', W'  feature-map size (60, 80)
# Primed names (u', v') are feature-map coordinates; unprimed are image pixels.

def mma(error: Tensor, tau):
    return torch.mean((error < tau).float())


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
        (uvs[..., 0] + 0.5) * (descriptors.shape[3] / images.shape[3]) - 0.5,
        (uvs[..., 1] + 0.5) * (descriptors.shape[2] / images.shape[2]) - 0.5,
    ], dim=-1)

    return bilinear_sample(descriptors, feature_uvs), descriptors


def descriptor_error(descriptors_i, descriptors_j, images_j, uvs_j):
    # (B, N, 2) as (u, v) to match uvs_j's layout
    predicted_uvs, _, _ = best_match(
        query_descriptors_i=descriptors_i,
        descriptors_j=descriptors_j,
        image_shape=(images_j.shape[2], images_j.shape[3])
    )
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
