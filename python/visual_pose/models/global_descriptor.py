import torch
from torch.nn import functional as F


def avg_pooling(descriptors: torch.Tensor):
    """
    Average all descriptors for an image into a single descriptor.

    :param descriptors: BxDxH'xW' descriptors for an image, B is always 1 for the prod pipeline

    :return: BxD normalized descriptors for images
    """
    return F.normalize(torch.mean(
        descriptors,
        dim=(-2,-1)
    ), dim=-1)


def gem_pooling(descriptors: torch.Tensor, p: int = 3):
    sum_desc = torch.sum(descriptors ** p, dim=(-2, -1))
    return F.normalize(torch.pow(sum_desc, 1.0 / p), dim=-1)