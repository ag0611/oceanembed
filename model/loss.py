import torch
import torch.nn as nn


DEPTH_WEIGHTS = torch.tensor([
    2.0,  # 0m
    2.0,  # 5m
    2.0,  # 10m
    2.0,  # 20m
    2.0,  # 30m
    1.5,  # 50m
    1.5,  # 75m
    1.0,  # 100m
    1.0,  # 125m
    1.0,  # 150m
    1.0,  # 200m
    1.0,  # 300m
    1.0,  # 500m
    1.0,  # 700m
    1.0,  # 1000m
], dtype=torch.float32)


class DepthWeightedHuberLoss(nn.Module):
    """
    Huber loss (delta=1.0) weighted by depth.
    Shallower depths get 2x weight (more oceanographically important).
    Input:  pred (B, 15), target (B, 15)
    Output: scalar loss
    """
    def __init__(self, delta=1.0):
        super().__init__()
        self.huber = nn.HuberLoss(reduction='none', delta=delta)
        self.register_buffer('weights', DEPTH_WEIGHTS)

    def forward(self, pred, target):
        loss = self.huber(pred, target)          # (B, 15)
        loss = loss * self.weights.unsqueeze(0)  # broadcast over batch
        return loss.mean()


if __name__ == '__main__':
    loss_fn = DepthWeightedHuberLoss(delta=1.0)
    pred    = torch.randn(4, 15)
    target  = torch.randn(4, 15)
    loss    = loss_fn(pred, target)
    print(f'Loss: {loss.item():.4f}')
    print('loss.py ready.')
