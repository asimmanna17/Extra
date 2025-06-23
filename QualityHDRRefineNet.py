import torch
import torch.nn as nn

# SE block (unchanged)
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        scale = self.se(x)
        return x * scale

# Dense layer inside RRDB
class DenseConv(nn.Module):
    def __init__(self, in_channels, growth_rate=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        out = self.conv(x)
        return torch.cat([x, out], dim=1)

# RRDB block
class RRDB(nn.Module):
    def __init__(self, channels, growth_rate=32, num_layers=3):
        super().__init__()
        modules = []
        in_ch = channels
        for _ in range(num_layers):
            modules.append(DenseConv(in_ch, growth_rate))
            in_ch += growth_rate
        self.dense_block = nn.Sequential(*modules)
        self.reduce = nn.Conv2d(in_ch, channels, 1)
    def forward(self, x):
        out = self.reduce(self.dense_block(x))
        return x + 0.2 * out  # residual scaled

# Main HDR enhancement network
class QualityHDRRefineNet(nn.Module):
    def __init__(self, in_channels=60, feat_channels=128, num_blocks=8):
        super().__init__()
        self.head = nn.Conv2d(in_channels, feat_channels, 3, padding=1)

        self.body = nn.Sequential(*[
            nn.Sequential(
                RRDB(feat_channels),
                SEBlock(feat_channels)
            ) for _ in range(num_blocks)
        ])

        self.tail = nn.Sequential(
            nn.Conv2d(feat_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, f2):
        residual = self.head(x)
        residual = self.body(residual)
        residual = self.tail(residual)
        out = x + residual + f2
        return self.final(out)
