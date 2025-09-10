from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class ProjectionDims:
    vision: int = 256
    omics: int = 256


class VisionEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.proj = nn.Linear(32, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x).flatten(1)
        return nn.functional.normalize(self.proj(x), dim=-1)


class OmicsEncoder(nn.Module):
    def __init__(self, in_dim: int = 1024, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU(), nn.Linear(512, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.net(x), dim=-1)


class ContrastiveAligner(nn.Module):
    def __init__(self, temp: float = 0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / temp))

    def forward(self, v: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        logits = self.logit_scale.exp() * v @ o.t()
        targets = torch.arange(v.size(0), device=v.device)
        loss = nn.functional.cross_entropy(logits, targets) + nn.functional.cross_entropy(logits.t(), targets)
        return loss

