import torch
import math

def positional_encoding_2d(h, w, d_dim, device):
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