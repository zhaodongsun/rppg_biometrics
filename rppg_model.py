import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from utils_sig import *

class rppg_model(nn.Module):
    def __init__(self, fs, in_channels=3, out_channels=1):
        super(rppg_model, self).__init__()
        self.fs = fs

        self.inc = (DoubleConv(in_channels, 32))

        self.down1 = (DownS(32, 64))
        self.down2 = (DownS(64, 128))
        self.down3 = (DownT(128, 256))
        self.down4 = (DownT(256, 512))

        self.up1 = (UpT(512, 256))
        self.up2 = (UpT(256, 128))
        self.outc = (OutConv(128, out_channels))

    def forward(self, x):
        # (B, 3, N, T)
        means = torch.mean(x, dim=(2, 3), keepdim=True)
        stds = torch.std(x, dim=(2, 3), keepdim=True)
        x = (x - means) / stds # (B, 3, N, T)
        
        x = self.inc(x) # (B, C, 36, 600)

        x = self.down1(x) # (B, C, 18, 600)
        x = self.down2(x) # (B, C, 9, 600)
        x = self.down3(x) # (B, C, 9, 300)
        x = self.down4(x) # (B, C, 9, 150)


        x = self.up1(x) # (B, C, 9, 300)
        x = self.up2(x) # (B, C, 9, 600)

        x = self.outc(x) # (2, 1, 4, 600)
        x = x[:, 0] # (2, 4, 600)

        # filtering
        filter_b, filter_a = butter_ba(lowcut=40/60, highcut=250/60, fs=self.fs)
        filter_a = torch.tensor(filter_a.astype('float32')).to(x.device)
        filter_b = torch.tensor(filter_b.astype('float32')).to(x.device)

        x = torchaudio.functional.filtfilt(x, filter_a, filter_b, clamp=False) # (2, 4, 600)
        return x, torch.mean(x, 1) # (2, 4, 600), (2, 600)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownS(nn.Module):
    """Downscaling with avgpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool2d((2, 1)),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.avgpool_conv(x)


class DownT(nn.Module):
    """Downscaling with avgpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool2d((1, 2)),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.avgpool_conv(x)


class UpT(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=(1, 2))
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.avg_down = nn.AvgPool2d((2, 1))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.avg_down(x)
        return self.conv(x)
 