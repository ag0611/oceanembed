import torch
import torch.nn as nn

DEPTHS = torch.tensor(
    [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000],
    dtype=torch.float32
)


class ThermoclineHead(nn.Module):
    """
    Auxiliary head: predicts thermocline depth from T[15] profile.
    Thermocline = depth of maximum dT/dz (steepest temperature gradient).

    Two outputs:
      thermo_depth : scalar (m)  -- soft-argmax over depth bins
      thermo_grad  : scalar      -- max gradient magnitude
    """
    def __init__(self, num_depths=15):
        super().__init__()
        self.register_buffer('depths', DEPTHS)

        self.grad_net = nn.Sequential(
            nn.Linear(num_depths - 1, 64),
            nn.ReLU(),
            nn.Linear(64, num_depths - 1)   # gradient at each layer boundary
        )

    def forward(self, T_pred):
        """
        T_pred: (B, 15)
        returns:
          thermo_depth (B,)  in metres
          thermo_grad  (B,)  deg-C / m
        """
        # Finite difference gradient: dT between consecutive depths
        dT = T_pred[:, 1:] - T_pred[:, :-1]                  # (B, 14)
        dZ = (self.depths[1:] - self.depths[:-1]).unsqueeze(0) # (1, 14)
        grad = dT / dZ                                         # (B, 14) dT/dz

        # Learned refinement of raw gradient
        grad_refined = self.grad_net(grad)                     # (B, 14)

        # Soft-argmax to get differentiable thermocline depth
        weights      = torch.softmax(grad_refined.abs(), dim=1)  # (B, 14)
        depth_bins   = ((self.depths[:-1] + self.depths[1:]) / 2).unsqueeze(0)
        thermo_depth = (weights * depth_bins).sum(dim=1)       # (B,)
        thermo_grad  = (weights * grad.abs()).sum(dim=1)        # (B,)

        return thermo_depth, thermo_grad


if __name__ == '__main__':
    head   = ThermoclineHead(num_depths=15)
    T_pred = torch.randn(4, 15)
    depth, grad = head(T_pred)
    print(f'Thermo depth: {depth}')
    print(f'Thermo grad:  {grad}')
    print('thermocline_head.py ready.')
