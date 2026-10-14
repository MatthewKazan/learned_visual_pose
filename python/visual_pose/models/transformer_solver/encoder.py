from typing import Optional

from torch import nn, Tensor


class EncoderCell(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            ff_dim: int,
            dropout: float,
            num_heads: int,
    ):
        super(EncoderCell, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x_ln = self.layer_norm1(x)
        y = x + self.self_attention(x_ln, x_ln, x_ln, mask, need_weights=False)[0]
        y_ln = self.layer_norm2(y)
        y = y + self.feed_forward(y_ln)
        return y


class Encoder(nn.Module):
    def __init__(self, num_layers: int, embed_dim: int, ff_dim: int, dropout: float, num_heads: int):
        super(Encoder, self).__init__()
        self.num_layers = num_layers

        self.encoder_cells = nn.ModuleList([
            EncoderCell(embed_dim, ff_dim, dropout, num_heads)
            for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        for cell in self.encoder_cells:
            x = cell(x, mask)
        return self.layer_norm(x)