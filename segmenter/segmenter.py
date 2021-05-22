from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskTransformer(nn.Module):

    def __init__(
        self,
        num_classes: int,
        emb_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
    ):
        self.num_classes = num_classes
        self.cls_tokens = nn.Parameter(torch.randn(1, num_classes, emb_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b = x.shape[0]

        # Concatenate class tokens
        cls_tokens = self.cls_tokens.repeat(b, 1, 1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = self.transformer(x)

        # Separate class tokens
        z = x[:, :-self.num_classes, :]
        c = x[:, -self.num_classes:, :]

        return z, c


class Segmenter(nn.Module):

    def __init__(
        self,
        num_classes,
        image_size,
        patch_size,
        emb_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
    ):
        self.encoder = None
        self.mask_transformer = MaskTransformer(
            num_classes,
            emb_dim,
            hidden_dim,
            num_layers,
            num_heads,
        )

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        z, c = self.mask_transformer(x)

        # Scalar product 
        torch.einsum("bik, bjk -> bji", c, z)
        return mask