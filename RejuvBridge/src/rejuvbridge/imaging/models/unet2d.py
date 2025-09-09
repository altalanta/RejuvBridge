from typing import List
import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True)
    )


class UNet2D(nn.Module):
    def __init__(self, in_ch: int = 3, n_classes: int = 1, features: List[int] = [64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        chs = in_ch
        for f in features:
            self.downs.append(conv_block(chs, f))
            chs = f
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(features[-1], features[-1]*2)
        rev = list(reversed(features))
        chs = features[-1]*2
        for f in rev:
            self.ups.append(nn.ConvTranspose2d(chs, f, 2, stride=2))
            self.ups.append(conv_block(chs, f))
            chs = f
        self.head = nn.Conv2d(chs, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i//2]
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i+1](x)
        return self.head(x)

