from typing import Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

import segmenter.vision_transformer


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
        cls_tokens = self.cls_tokens.repeat(b, 1, 1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.transformer(x)
        c = x[:, :self.num_classes]
        z = x[:, self.num_classes:]
        return z, c


class Upsample(nn.Module):

    def __init__(self, image_size: int, patch_size: Tuple[int, int]):
        self.model = nn.Sequential(
            Rearrange("b (p1 p2) c -> b c p1 p2", p1=image_size//patch_size[0], p2=image_size//patch_size[1]),
            nn.Upsample(scale_factor=patch_size, mode="bilinear")
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Segmenter(nn.Module):

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        image_size: int,
        emb_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
    ):
        self.encoder = timm.create_model(backbone, img_size=image_size, pretrained=True)
        patch_size = self.encoder.patch_embed.patch_size
        self.mask_transformer = MaskTransformer(
            num_classes,
            emb_dim,
            hidden_dim,
            num_layers,
            num_heads,
        )
        self.upsample = Upsample(image_size, patch_size)
        self.scale = emb_dim ** -0.5

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        z, c = self.mask_transformer(x)
        masks = z @ c.transpose(1, 2)
        masks = torch.softmax(masks / self.scale, dim=-1)
        return self.upsample(masks)
