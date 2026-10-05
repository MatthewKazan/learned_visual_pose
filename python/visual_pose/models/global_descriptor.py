import torch
from torch.nn import functional as F
from torch import nn, Tensor

from visual_pose.models.positional_encoding import positional_encoding_2d


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


class AttentionPooling(nn.Module):
    def __init__(self, descriptor_dim: int, atten_ch: int = 256, kernel_size: int = 4, stride: int = 4, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(descriptor_dim, atten_ch, kernel_size, stride,
                                          padding=0, dilation=dilation, bias=False)
        self.w_key = nn.Linear(atten_ch, atten_ch, bias=False)
        self.w_value = nn.Linear(atten_ch, atten_ch, bias=False)
        self.w_query = nn.Linear(atten_ch, atten_ch, bias=False)
        self.output_layer = nn.Linear(atten_ch, atten_ch, bias=False)
        # Start as average pooling of the input
        with torch.no_grad():
            self.conv.weight.zero_()
            channels = torch.arange(min(descriptor_dim, atten_ch))
            self.conv.weight[channels, channels] = 1.0 / (kernel_size * kernel_size)
            if self.conv.bias is not None:
                self.conv.bias.zero_()
            nn.init.zeros_(self.w_query.weight)
            self.w_value.weight.copy_(torch.eye(atten_ch))
            self.output_layer.weight.copy_(torch.eye(atten_ch))
        self.atten_ch = atten_ch
        self.pe_scalar = nn.Parameter(torch.full((1,), 0.1))


    def forward(self, x: Tensor):
        x = self.conv(x)
        pe = positional_encoding_2d(x.shape[2], x.shape[3], self.atten_ch, x.device) * self.pe_scalar
        pe = pe.permute(2, 0, 1) # [C, H, W]


        x = x + pe.unsqueeze(0)  # [B, C, H, W]
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat([x, x.mean(dim=1, keepdim=True)], dim=1)

        q = self.w_query(x[:,-1:,:])
        k = self.w_key(x[:, :-1, :])
        v = self.w_value(x[:, :-1, :])

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.atten_ch ** .5)
        a = torch.softmax(scores, dim=-1)
        pooled = torch.matmul(a,v).squeeze(1)
        return F.normalize(self.output_layer(pooled), dim=-1)



