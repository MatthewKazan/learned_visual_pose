"""
Reading descriptors out of a feature map, and scoring how well they match.

Used by both training and evaluation. Nothing here imports a model class, so
SIFT goes through the same metric as the CNN.
"""
import numpy as np
import torch
from torch import floor, nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader

from visual_pose.data_utils.constants import DEVICE
from visual_pose.data_utils.loaders import on_device
from visual_pose.geometry.best_match import best_match


# B pairs, N correspondences, D descriptor dim, M = H'*W' candidates.
# Primed (u', v') is feature-map coordinates; unprimed is image pixels.

def mma(error: Tensor, tau: float) -> Tensor:
    """Fraction of errors under tau. error (N,) pixels -> 0-d tensor."""
    return torch.mean((error < tau).float())


def bilinear_sample_descriptor(descriptor: Tensor, feature_uvs: Tensor) -> Tensor:
    """
    descriptor (B,D,H',W'), feature_uvs (B,N,2) in feature coords -> (B,N,D).

    A projected pixel lands between cells, so blend the four surrounding
    descriptors by area weight.
    """
    u0 = floor(feature_uvs[...,0])   # (B, N)
    v0 = floor(feature_uvs[...,1])   # (B, N)

    # (B,C,H,W) -> (B,H,W,C) so a descriptor is a trailing slice and the
    # gather below returns (B, N, D) directly
    descriptor = descriptor.permute(0, 2, 3, 1)

    # fractional parts, unsqueezed to (B, N, 1) so they broadcast across D
    a = (feature_uvs[..., 0] - u0).unsqueeze(-1)
    b = (feature_uvs[..., 1] - v0).unsqueeze(-1)

    u1 = (u0 + 1).clamp(0, descriptor.shape[2] - 1)
    v1 = (v0 + 1).clamp(0, descriptor.shape[1] - 1)
    u0 = u0.clamp(0, descriptor.shape[2] - 1)
    v0 = v0.clamp(0, descriptor.shape[1] - 1)

    # (B, 1) to broadcast against the (B, N) coords; without it the gather
    # returns the cross product (B, D, B, N) instead of the diagonal
    batch = torch.arange(descriptor.shape[0]).unsqueeze(-1).to(descriptor.device)

    # each term: (B,N,1) weight * (B,N,D) corner -> (B, N, D)
    w1 = (1 - a) * (1 - b) * descriptor[batch, v0.long(), u0.long()]
    w2 = (1 - a) * b *       descriptor[batch, v1.long(), u0.long()]
    w3 = a * (1 - b) *       descriptor[batch, v0.long(), u1.long()]
    w4 = a * b *             descriptor[batch, v1.long(), u1.long()]

    # blending unit vectors is not unit length
    return F.normalize(w1 + w2 + w3 + w4, dim=-1)


def descriptors_at(model: nn.Module, images: Tensor,
                   uvs: Tensor) -> tuple[Tensor, Tensor]:
    """
    Run the model and read descriptors at given image pixels.

    images (B,3,H,W), uvs (B,N,2) in image pixels -> (sampled (B,N,D),
    feature_map (B,D,H',W')). The map comes back too because eval needs the
    whole thing as a candidate pool.
    """
    descriptors = model(images)

    # u scales by width, v by height. Crossing them is silent when the strides
    # are equal, so it lives in one place. Must match cells_to_pixels.
    feature_uvs = torch.stack([
        (uvs[..., 0] + 0.5) * (descriptors.shape[3] / images.shape[3]) - 0.5,
        (uvs[..., 1] + 0.5) * (descriptors.shape[2] / images.shape[2]) - 0.5,
    ], dim=-1)

    return bilinear_sample_descriptor(descriptors, feature_uvs), descriptors


def descriptor_error(descriptors_i: Tensor, descriptors_j: Tensor,
                     images_j: Tensor, uvs_j: Tensor) -> tuple[Tensor, Tensor]:
    """
    Pixel error of each query's best match against the truth.

    descriptors_i (B, N, D) queries, descriptors_j (B, D, H', W') the pool,
    images_j (B, 3, H, W) for the pixel scale, uvs_j (B, N, 2) the answers.
    Returns (pixel_error (B, N), offset (B, N, 2) signed, as (du, dv)).
    """
    predicted_uvs, _, _ = best_match(
        query_descriptors_i=descriptors_i,
        descriptors_j=descriptors_j,
        image_shape=(images_j.shape[2], images_j.shape[3])
    )
    # signed, so the mean should sit near 0. A consistent +-4 means a half-cell
    # convention error, which MMA alone reads as merely "mediocre".
    offset = predicted_uvs - uvs_j
    pixel_error = torch.linalg.vector_norm(offset, dim=-1)
    return pixel_error, offset

def test_model(model: nn.Module, data_loader: DataLoader,
               taus: tuple[float, ...] = (1, 3, 4, 8, 12, 16),
               identity: bool = False) -> Tensor:
    """
    Per-correspondence pixel error over the whole loader, (total,).

    Caller applies mma() at whatever tau; `taus` only affects what is printed.

    identity=True matches each image against itself, so the right answer is
    "the nearest grid cell" and MMA should hit the quantization ceiling. Any
    shortfall there is a coordinate bug rather than a weak descriptor, which a
    normal run cannot distinguish.
    """
    model.to(DEVICE)
    # set the model in evaluation mode so the batch norm layers will behave correctly
    model.eval()

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

            sampled_descriptors_i, descriptors_i = descriptors_at(model, images_i, uvs_i)
            descriptors_j = model(images_j)   # (B, D, H', W')

            pixel_error, offset = descriptor_error(sampled_descriptors_i, descriptors_j, images_j, uvs_j)
            errors.append(pixel_error.flatten())
            offsets.append(offset.flatten(0, 1))
            stride = images_j.shape[3] / descriptors_j.shape[3]

        errors = torch.cat(errors)             # (total_correspondences,)
        offsets = torch.cat(offsets)           # (total_correspondences, 2)
        H, W = images_i.shape[2], images_i.shape[3]

        # inliers only: the outlier tail dominates the raw mean and buries a
        # real +-4 bias
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
            # random guess: pi*tau^2 / (H*W). Ceiling: the fraction of one
            # cell within tau of its own centre.
            floor_ = np.pi * tau ** 2 / (H * W)
            ceiling = min(1.0, np.pi * tau ** 2 / stride ** 2)
            print(f"{tau:>5} {mma(errors, tau):>8.2%} {floor_:>9.3%} {ceiling:>8.1%}")
        return errors
