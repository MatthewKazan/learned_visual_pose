"""
Frozen DINOv2 as a per-patch semantic descriptor. LEARN FIRST (VIS-058):
this is the wrapper the generator needs to run, not a design commitment.
"""
import torch
import torch.nn.functional as F

from visual_pose.geometry.matching import bilinear_sample_descriptor
from visual_pose.config.constants import DEVICE

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
        """x (B, 3, H, W) float in [0, 1] -> (B, D, H//14, W//14) patch descriptors, L2-normalised."""
        return self.with_cls(x)[0]

    @torch.no_grad()
    def with_cls(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x (B, 3, H, W) float in [0, 1] -> patch grid as __call__, and the CLS token (B, D)
        as DINOv2 returns it (final LayerNorm, NOT unit length: normalising is a choice
        left to whoever uses it).

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
        features = self.model.forward_features((x - mean) / std)
        patch = features["x_norm_patchtokens"]   # (B, h/14 * w/14, D)
        grid = patch.reshape(x.shape[0], h // PATCH, w // PATCH, self.dim).permute(0, 3, 1, 2)
        return F.normalize(grid, dim=1), features["x_norm_clstoken"]

    def descriptors_for_images(self, rgb: torch.Tensor, uvs: torch.Tensor,
                               batch_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
        """x (B, 3, H, W) float in [0, 1] -> (B, D, H//14, W//14) patch descriptors, L2-normalised."""
        H, W = rgb.shape[-3:-1]
        h, w = H - H % PATCH, W - W % PATCH
        top, left = (H - h) // 2, (W - w) // 2
        descriptors, cls = [], []
        with torch.inference_mode():
            for b in range(0, len(rgb), batch_size):
                x = rgb[b:b + batch_size].to(DEVICE).permute(0, 3, 1, 2).float().div_(255)
                raw_descriptors, batch_cls = self.with_cls(x)
                cls.append(batch_cls.cpu())
                batch_uvs = uvs[b:b + batch_size]
                feature_uvs = torch.stack([
                    ((batch_uvs[..., 0] - left) + 0.5) * (raw_descriptors.shape[3] / w) - 0.5,
                    ((batch_uvs[..., 1] - top) + 0.5) * (raw_descriptors.shape[2] / h) - 0.5,
                ], dim=-1)
                descriptors.append(bilinear_sample_descriptor(raw_descriptors, feature_uvs).cpu())

        return torch.cat(descriptors, dim=0), torch.cat(cls, dim=0)


