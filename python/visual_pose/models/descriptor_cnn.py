from torch import nn, Tensor
from torch.nn import functional as F
from typing import List

from visual_pose.config import Config


class DescriptorCNN(nn.Module):
    def __init__(
        self,
        body_channels: List[int]     = [64, 128, 256],
        body_kernel_sizes: list[int] = [3, 3, 3],
        body_strides: list[int]      = [2, 2, 2],
        body_dilations: list[int]    = None,
        descriptor_dim: int = 128,
        norm: str = "batch",
    ):
        super().__init__()

        n = len(body_channels)
        # Dilation spreads the taps instead of adding them, so the span is
        # (k-1)*d + 1 at fixed parameters. RF 15 -> 57 was +40pp, and kernels
        # pay for reach quadratically where dilation pays nothing.
        body_dilations = [1] * n if body_dilations is None else list(body_dilations)

        assert len(body_kernel_sizes) == n, "body_channels and body_kernel_sizes must match"
        assert len(body_strides)      == n, "body_channels and body_strides must match"
        assert len(body_dilations)    == n, "body_channels and body_dilations must match"

        self.descriptor_dim = descriptor_dim
        self.body_channels = body_channels
        self.norm = norm

        # backbone: stem + body, all Conv → norm → ReLU
        blocks = []
        in_ch = 3   # RGB input
        for out_ch, k, s, d in zip(body_channels, body_kernel_sizes, body_strides, body_dilations):
            blocks.append(self._conv_block(in_ch, out_ch, kernel_size=k, stride=s,
                                           dilation=d, norm=norm))
            in_ch = out_ch
        self.backbone = nn.Sequential(*blocks)

        # head: 1×1 conv to descriptor_dim, no BN/ReLU (F.normalize does the rest)
        self.head = nn.Conv2d(body_channels[-1], descriptor_dim, kernel_size=1)

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int, kernel_size: int, stride: int,
                    dilation: int = 1, norm: str = "batch") -> nn.Sequential:
        # BatchNorm blends statistics across a shuffled multi-environment
        # batch; GroupNorm is per-sample and immune.
        if norm == "batch":
            norm_layer = nn.BatchNorm2d(out_ch)
        elif norm == "group":
            norm_layer = nn.GroupNorm(min(32, out_ch), out_ch)
        elif norm == "none":
            norm_layer = nn.Identity()
        else:
            raise ValueError(f"unknown norm {norm!r}, expected batch|group|none")

        # 'same' padding is half the EFFECTIVE kernel: dilation * (k-1)/2.
        # k//2 shrinks the map silently, which breaks every coordinate
        # conversion and reads as "the descriptor got worse".
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride,
                      padding=dilation * (kernel_size - 1) // 2, dilation=dilation),
            norm_layer,
            nn.ReLU(inplace=True),
        )

    @classmethod
    def from_config(cls, cfg: Config) -> "DescriptorCNN":
        return cls(
            body_channels=cfg.body_channels,
            body_kernel_sizes=cfg.body_kernel_sizes,
            body_strides=cfg.body_strides,
            body_dilations=cfg.body_dilations,
            descriptor_dim=cfg.descriptor_dim,
            norm=cfg.norm,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, 3, H, W) RGB image, values in [0, 1]
        returns: (B, descriptor_dim, H', W') L2-normalized descriptors,
                 where H' = H / prod(strides), W' = W / prod(strides).
        """
        x = self.backbone(x)
        x = self.head(x)
        return F.normalize(x, dim=1)
