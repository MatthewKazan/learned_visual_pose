from torch import nn
from torch.nn import functional as F
from typing import List


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
        # Dilation spreads a kernel's taps apart instead of adding taps, so the
        # span grows to (k-1)*d + 1 for the same weights and FLOPs. Receptive
        # field is the one axis that has moved the metric (RF 15 -> 57 was
        # +40pp), and bigger kernels pay for it quadratically: kernel 9 reaches
        # RF 57 for 3.4M params where dilation (1,3,9) reaches RF 87 for 0.4M.
        #
        # It also isolates the variable. Depth or width would grow reach AND
        # capacity together; dilation holds parameters, FLOPs and layer count
        # fixed, so a change in the metric can only be reach.
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
        # BatchNorm normalizes using statistics across the batch. With shuffled
        # multi-environment batches one batch can hold a bright hospital and a
        # near-black night factory, so those statistics blend unrelated scenes.
        # GroupNorm is per-sample and immune to that.
        if norm == "batch":
            norm_layer = nn.BatchNorm2d(out_ch)
        elif norm == "group":
            norm_layer = nn.GroupNorm(min(32, out_ch), out_ch)
        elif norm == "none":
            norm_layer = nn.Identity()
        else:
            raise ValueError(f"unknown norm {norm!r}, expected batch|group|none")

        # 'same' padding is half the EFFECTIVE kernel, so dilation scales it:
        # (k_eff - 1)/2 = dilation * (k-1)/2. Leaving it at k//2 shrinks the map
        # by 2*(d-1) per layer, which does not raise -- it silently changes the
        # descriptors.shape/images.shape ratio that every coordinate conversion
        # is derived from, and reads as "the descriptor got worse".
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride,
                      padding=dilation * (kernel_size - 1) // 2, dilation=dilation),
            norm_layer,
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        x: (B, 3, H, W) RGB image, values in [0, 1]
        returns: (B, descriptor_dim, H', W') L2-normalized descriptors,
                 where H' = H / prod(strides), W' = W / prod(strides).
        """
        x = self.backbone(x)
        x = self.head(x)
        return F.normalize(x, dim=1)
