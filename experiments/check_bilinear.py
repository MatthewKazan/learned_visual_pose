"""
Scratch check: compare bilinear_sample against F.grid_sample.

Run: python -m experiments.check_bilinear
"""
import torch
from torch.nn import functional as F

from visual_pose.matching import bilinear_sample_descriptor


def reference(descriptor, feature_uvs):
    """F.grid_sample equivalent. descriptor (B,D,H,W), feature_uvs (B,N,2) in feature coords."""
    B, D, H, W = descriptor.shape
    u = feature_uvs[..., 0]
    v = feature_uvs[..., 1]
    # feature coords -> normalized [-1,1], align_corners=True convention
    grid = torch.stack([2 * u / (W - 1) - 1, 2 * v / (H - 1) - 1], dim=-1)  # (B,N,2)
    out = F.grid_sample(descriptor, grid.unsqueeze(2), mode="bilinear", align_corners=True)
    out = out.squeeze(-1).permute(0, 2, 1)  # (B,N,D)
    return F.normalize(out, dim=-1)  # bilinear_sample normalizes, so match it


if __name__ == "__main__":
    torch.manual_seed(0)
    B, D, H, W, N = 2, 4, 8, 10, 5

    descriptor = torch.randn(B, D, H, W)
    # fractional coords kept away from the borders so clamping can't matter
    u = torch.rand(B, N) * (W - 2) + 0.5
    v = torch.rand(B, N) * (H - 2) + 0.5
    feature_uvs = torch.stack([u, v], dim=-1)  # (B,N,2)

    expected = reference(descriptor, feature_uvs)
    print(f"expected shape: {tuple(expected.shape)}")

    actual = bilinear_sample_descriptor(descriptor, feature_uvs)
    print(f"actual   shape: {tuple(actual.shape)}")

    print(f"\nexpected[0,0]: {expected[0, 0]}")
    print(f"actual  [0,0]: {actual[0, 0] if actual.shape == expected.shape else '<shape mismatch>'}")

    if actual.shape == expected.shape:
        max_err = (actual - expected).abs().max().item()
        print(f"\nmax abs error: {max_err:.2e}")
        print("MATCH" if torch.allclose(actual, expected, atol=1e-5) else "MISMATCH")
