import torch
import torch.nn as nn
import torch.nn.functional as F


class PeriodicConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, padding=0, **kwargs)
        self.do_pad = kernel_size > 1

    def forward(self, x):
        if self.do_pad:
            x = F.pad(x, (1, 1, 1, 1), mode="circular")
        return super().forward(x)


class UNetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            PeriodicConv2d(1, 16, 3),
            nn.ReLU(inplace=True),
            PeriodicConv2d(16, 16, 3),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            PeriodicConv2d(16, 32, 3),
            nn.ReLU(inplace=True),
            PeriodicConv2d(32, 32, 3),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            PeriodicConv2d(32, 64, 3),
            nn.ReLU(inplace=True),
            PeriodicConv2d(64, 64, 3),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(
            PeriodicConv2d(64, 32, 3),
            nn.ReLU(inplace=True),
            PeriodicConv2d(32, 32, 3),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = nn.Sequential(
            PeriodicConv2d(32, 16, 3),
            nn.ReLU(inplace=True),
            PeriodicConv2d(16, 16, 3),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)
