import torch
import torch.nn as nn
import math


class PatchEmbed(nn.Module):
    """
    Split (B, C, 64, 64) into 16x16 patches -> linear embedding.
    patch_size=16 -> 4x4 = 16 patches per image
    """
    def __init__(self, in_channels=5, patch_size=16, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, 64, 64) -> (B, embed_dim, 4, 4) -> (B, 16, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViTEncoder(nn.Module):
    """
    Vision Transformer encoder (ViT-B/16 style) for satellite patches.
    Input:  (B, 5, 64, 64)
    Output: (B, 512) latent embedding
    """
    def __init__(
        self,
        in_channels=5,
        patch_size=16,
        embed_dim=768,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        embedding_dim=512
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(in_channels, patch_size, embed_dim)
        num_patches = (64 // patch_size) ** 2  # 16 patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True        # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)                          # (B, 16, 768)

        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, 768)
        x   = torch.cat([cls, x], dim=1)                 # (B, 17, 768)
        x   = x + self.pos_embed                         # add positional

        x   = self.transformer(x)                        # (B, 17, 768)
        x   = self.norm(x)

        cls_out = x[:, 0]                                # (B, 768) CLS token
        z       = self.projection(cls_out)               # (B, 512)
        return z


if __name__ == '__main__':
    model = ViTEncoder(in_channels=5, embedding_dim=512)
    x     = torch.randn(4, 5, 64, 64)
    z     = model(x)
    print(f'Input:  {x.shape}')
    print(f'Output: {z.shape}')
    total = sum(p.numel() for p in model.parameters())
    print(f'Params: {total / 1e6:.2f}M')
