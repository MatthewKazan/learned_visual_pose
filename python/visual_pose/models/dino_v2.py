"""
Frozen DINOv2 as a per-patch semantic descriptor. LEARN FIRST (VIS-058):
this is the wrapper the generator needs to run, not a design commitment.
"""
import torch
import torch.nn.functional as F

from visual_pose.data_utils.constants import DEVICE

PATCH = 14   # ViT-S/14: H and W must be multiples of this
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SemanticDescriptor:
    def __init__(self, repo: str = "facebookresearch/dinov2", model: str = "dinov2_vits14_reg"):
        self.model = torch.hub.load(repo, model).eval().to(DEVICE)   # downloads on first use
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dim = self.model.embed_dim

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x (B, 3, H, W) float in [0, 1] -> (B, D, H//14, W//14) patch descriptors, L2-normalised.

        Centre-crops to a multiple of 14 first (480x640 -> 476x630, a 34x45
        grid); DINOv2 refuses other sizes. ImageNet normalisation is what the
        checkpoint was trained with and is not optional.
        """
        H, W = x.shape[-2:]
        h, w = H - H % PATCH, W - W % PATCH
        top, left = (H - h) // 2, (W - w) // 2
        x = x[..., top:top + h, left:left + w]
        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        patch = self.model.forward_features((x - mean) / std)["x_norm_patchtokens"]   # (B, h/14 * w/14, D)
        grid = patch.reshape(x.shape[0], h // PATCH, w // PATCH, self.dim).permute(0, 3, 1, 2)
        return F.normalize(grid, dim=1)
