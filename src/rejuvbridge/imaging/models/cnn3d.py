import torch
import torch.nn as nn


class CNN3D(nn.Module):
    """Tiny 3D CNN for Cryo-EM density classification (stub)."""

    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x).flatten(1)
        return self.fc(x)

