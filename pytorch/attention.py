"""
This code snippet shows an example of implementing Multi-Head Attention in PyTorch.

1. Split the input tensors into multiple heads.
2. Compute the scaled dot-product attention.
3. Concatenate the output tensors from multiple heads.
"""

from torch import nn, Tensor
from functools import partial


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention."""

    def __init__(self, d_model: int, n_heads: int):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.proj_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        """Forward pass of Multi-Head Attention."""
        q, k, v = self.proj_q(q), self.proj_k(k), self.proj_v(v)
        q, k, v = map(partial(_split, n_heads=self.n_heads), (q, k, v))
        out, attn = self.attention(q, k, v, mask)
        out = _concat(out)
        return self.proj_o(out)


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention."""

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None
    ) -> tuple[Tensor, Tensor]:
        """Forward pass of Scaled Dot-Product Attention."""
        d_k = k.size()[2]
        k_t = k.transpose(1, 2)
        scores: Tensor = (q @ k_t) * d_k**-0.5
        if mask is not None:
            scores.masked_fill_(mask, float("-inf"))
        attn = self.softmax(scores)
        out = attn @ v
        return out, attn  # attn for visualization


def _split(tensor: Tensor, n_heads: int) -> Tensor:
    """Split tensor by number of heads."""
    batch_size, seq_len, d_model = tensor.size()
    d_head = d_model // n_heads
    return tensor.view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)


def _concat(tensor: Tensor) -> Tensor:
    """Concatenate tensor."""
    batch_size, n_heads, seq_len, d_head = tensor.size()
    d_model = n_heads * d_head
    return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
