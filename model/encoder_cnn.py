import torch
import torch.nn as nn
import torchvision.models as models


class CNNEncoder(nn.Module):
    """
    ResNet-50 based CNN encoder for satellite surface observations.
    Input:  (B, 5, 64, 64) — 5 surface channels (SST, SSH, SSS, U, V)
    Output: (B, 512)       — latent embedding per spatial patch
    """

    def __init__(self, in_channels=5, embedding_dim=512):
        super().__init__()

        # Load ResNet-50, replace first conv for 5-channel input
        resnet = models.resnet50(weights=None)
        resnet.conv1 = nn.Conv2d(
            in_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove final FC + avgpool — keep feature extractor only
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # (B, 2048, 2, 2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))   # (B, 2048, 1, 1)

        self.projection = nn.Sequential(
            nn.Flatten(),                          # (B, 2048)
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, embedding_dim),        # (B, 512)
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x):
        """
        x: (B, 5, 64, 64)
        returns: z (B, 512)
        """
        feat = self.backbone(x)   # (B, 2048, 2, 2)
        feat = self.pool(feat)    # (B, 2048, 1, 1)
        z    = self.projection(feat)
        return z


if __name__ == '__main__':
    model = CNNEncoder(in_channels=5, embedding_dim=512)
    x     = torch.randn(4, 5, 64, 64)   # batch=4
    z     = model(x)
    print(f'Input:  {x.shape}')
    print(f'Output: {z.shape}')
    total = sum(p.numel() for p in model.parameters())
    print(f'Params: {total / 1e6:.2f}M')
