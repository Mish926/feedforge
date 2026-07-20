"""BERT4Rec, implemented from the paper (Sun et al., CIKM 2019).

A bidirectional transformer encoder over item-ID sequences, trained with
a cloze (masked language model) objective: random items in the sequence
are replaced with [MASK] and the model predicts them from both left and
right context. At inference, [MASK] is appended after the user's history
and the model's distribution over items at that position is the
next-item prediction.

Implementation choices worth defending in review:
- Learned positional embeddings (as in the paper), sized to max_len.
- Output layer shares weights with the item embedding matrix plus a
  per-item bias. Weight tying cuts parameters roughly in half at this
  scale and is what the paper does.
- GELU activations and pre-norm-free encoder blocks via PyTorch's
  TransformerEncoder with batch_first, which matches the original
  architecture closely enough that published hyperparameters transfer.
- Padding positions are excluded from attention via key padding mask.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .data import PAD


class BERT4Rec(nn.Module):
    def __init__(
        self,
        vocab_size: int,          # n_items + 2 (PAD + MASK)
        max_len: int = 200,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_len = max_len
        self.item_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_norm = nn.LayerNorm(d_model)
        self.emb_dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_bias = nn.Parameter(torch.zeros(vocab_size))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.item_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[PAD].zero_()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, L) -> logits over vocab: (B, L, V)"""
        B, L = tokens.shape
        positions = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(tokens) + self.pos_emb(positions)
        x = self.emb_dropout(self.emb_norm(x))

        pad_mask = tokens == PAD  # True where attention should ignore
        h = self.encoder(x, src_key_padding_mask=pad_mask)

        # Tied output projection: (B, L, D) @ (D, V) + bias
        logits = h @ self.item_emb.weight.T + self.out_bias
        return logits

    @torch.no_grad()
    def score_last_position(self, tokens: torch.Tensor) -> torch.Tensor:
        """Scores over the vocabulary at the final (MASK) position.

        tokens: (B, L) built by data.inference_batch -> (B, V)
        """
        logits = self.forward(tokens)
        return logits[:, -1, :]
