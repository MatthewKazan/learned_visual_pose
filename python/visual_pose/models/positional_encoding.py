import torch
import math

from torch import nn

from visual_pose.data_utils.constants import DEVICE


def positional_encoding_2d(h, w, d_dim, device = DEVICE):
    assert d_dim % 4 == 0

    d_axis = d_dim // 2

    div_term = torch.exp(
        torch.arange(0, d_axis, 2, device=device)
        * (-math.log(10000.0) / d_axis)
    )

    y = torch.arange(h, device=device).float()[:, None]
    x = torch.arange(w, device=device).float()[:, None]

    pe_y = torch.zeros(h, d_axis, device=device)
    pe_x = torch.zeros(w, d_axis, device=device)

    pe_y[:, 0::2] = torch.sin(y * div_term)
    pe_y[:, 1::2] = torch.cos(y * div_term)

    pe_x[:, 0::2] = torch.sin(x * div_term)
    pe_x[:, 1::2] = torch.cos(x * div_term)

    pe = torch.zeros(h, w, d_dim, device=device)

    pe[:, :, :d_axis] = pe_y[:, None, :]
    pe[:, :, d_axis:] = pe_x[None, :, :]

    return pe


def positional_encoding_1d(seq_len: int, input_dim: int, max_len: int = 10000, device = DEVICE):
        # PE_{(pos,2i)} = sin(pos / 10000^{2i/\dmodel}) \\                        #
        # PE_{(pos,2i+1)} = cos(pos / 10000^{2i/\dmodel})                         #

        pe = torch.zeros(seq_len, input_dim)
        denom = max_len ** (torch.arange(0, input_dim, 2) / input_dim)
        # Works because of pytorch broadcasting, need unssqueeze for broadcasting to work
        pe[:, 0::2] = torch.sin(torch.arange(seq_len).unsqueeze(1) / denom)
        pe[:, 1::2] = torch.cos(torch.arange(seq_len).unsqueeze(1) / denom)

        return pe.to(device)
