import torch
import torch.nn as nn


class SkipMLP(nn.Module):
    """Single MLP block with skip connection."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class SubsurfaceDecoder(nn.Module):
    """
    MLP decoder with skip connections.
    Input:  (B, 1024) — fused CNN+ViT embedding
    Output: (B, 15)   — temperature at 15 standard depth levels (C)
    """

    def __init__(self, embedding_dim=1024, hidden_dim=512, num_depths=15):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.blocks = nn.ModuleList([
            SkipMLP(hidden_dim),
            SkipMLP(hidden_dim),
            SkipMLP(hidden_dim),
        ])

        # Depth-aware output: each depth gets its own head
        self.depth_head = nn.Linear(hidden_dim, num_depths)

    def forward(self, z):
        """
        z: (B, 1024)
        returns: T (B, 15) — predicted temperature at each depth
        """
        x = self.input_proj(z)
        for block in self.blocks:
            x = block(x)
        T = self.depth_head(x)
        return T


class DualEncoderWithDecoder(nn.Module):
    """
    Full model: CNN + ViT encoders fused -> decoder -> T[15 depths]
    CNN output (B,512) + ViT output (B,512) -> concat (B,1024) -> decoder
    """
    def __init__(self, cnn_encoder, vit_encoder, num_depths=15):
        super().__init__()
        self.cnn = cnn_encoder
        self.vit = vit_encoder
        self.decoder = SubsurfaceDecoder(
            embedding_dim=1024,
            hidden_dim=512,
            num_depths=num_depths
        )

    def forward(self, x):
        """
        x: (B, 5, 64, 64)
        returns: T (B, 15)
        """
        z_cnn = self.cnn(x)              # (B, 512)
        z_vit = self.vit(x)              # (B, 512)
        z     = torch.cat([z_cnn, z_vit], dim=1)   # (B, 1024)
        T     = self.decoder(z)          # (B, 15)
        return T


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from encoder_cnn import CNNEncoder
    from encoder_vit import ViTEncoder

    cnn = CNNEncoder(in_channels=5, embedding_dim=512)
    vit = ViTEncoder(in_channels=5, embedding_dim=512)

    model = DualEncoderWithDecoder(cnn, vit, num_depths=15)
    x     = torch.randn(4, 5, 64, 64)
    T     = model(x)

    print(f'Input:  {x.shape}')
    print(f'Output: {T.shape}')
    total = sum(p.numel() for p in model.parameters())
    print(f'Total params: {total / 1e6:.2f}M')
    print(f'Output range: {T.min().item():.2f} to {T.max().item():.2f} C (random weights)')
