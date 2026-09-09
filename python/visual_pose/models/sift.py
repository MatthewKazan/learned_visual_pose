import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SIFT(nn.Module):
    """OpenCV SIFT as an nn.Module, on the same grid as the CNN.

    Descriptors are computed at fixed grid keypoints rather than detected ones,
    so the comparison holds the sampling constant and varies only the
    descriptor. Output (B, 128, H', W'), L2-normalised over the channel.
    """

    # SIFT's patch diameter. Unlike its own detections (which set size from
    # scale-space), supplied keypoints need one chosen. 16 is closest to the
    # CNN's 15px receptive field, so neither model sees more context.
    DEFAULT_KEYPOINT_SIZE = 16.0

    def __init__(self, H: int = 480, W: int = 640, step: int = 8,
                 keypoint_size: float = DEFAULT_KEYPOINT_SIZE,
                 angle: float = -1.0):
        super().__init__()
        self.H, self.W, self.step = H, W, step
        self.Hf, self.Wf = H // step, W // step
        self.descriptor_dim = 128          # SIFT is fixed at 128
        self.sift = cv2.SIFT_create()

        # Row-major: v outer, u inner, so grid[a*Wf + b] is (u=b*step, v=a*step).
        # This MUST match the reshape in forward() or the feature map comes out
        # transposed -- which does not raise, it just looks like a bad descriptor.
        self.grid = [
            cv2.KeyPoint(float(b * step), float(a * step), keypoint_size, angle)
            for a in range(self.Hf) for b in range(self.Wf)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maps = []
        for img in x:                                  # (3, H, W) float [0,1]
            # OpenCV wants HWC uint8; SIFT specifically requires 8-bit input
            hwc = (img.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            grey = cv2.cvtColor(hwc, cv2.COLOR_RGB2GRAY)

            kps_out, desc = self.sift.compute(grey, self.grid)
            # compute() can drop keypoints it cannot describe. If it does, desc
            # no longer aligns with grid positions and the reshape below would
            # silently scramble the map.
            assert len(kps_out) == len(self.grid), (
                f"SIFT returned {len(kps_out)} of {len(self.grid)} keypoints; "
                f"descriptors no longer align with grid positions")

            # (M, 128) -> (Hf, Wf, 128) -> (128, Hf, Wf)
            maps.append(torch.from_numpy(desc).reshape(self.Hf, self.Wf, -1)
                        .permute(2, 0, 1))

        out = torch.stack(maps).to(x.device)
        # raw SIFT values run 0-207; the eval path assumes unit vectors so that
        # a dot product IS the cosine
        return F.normalize(out, dim=1)


if __name__ == "__main__":
    m = SIFT()
    x = torch.rand(2, 3, 480, 640)
    y = m(x)
    print(f"input:  {tuple(x.shape)}")
    print(f"output: {tuple(y.shape)}   (expect (2, 128, 60, 80))")
    print(f"desc norm at (0, :, 0, 0): {y[0, :, 0, 0].norm().item():.4f}")
    print(f"params: {sum(p.numel() for p in m.parameters())}  (SIFT learns nothing)")
