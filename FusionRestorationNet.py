import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# ConvNeXt V2 Block (CN)
# ----------------------------
class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, dim * 4, kernel_size=1)
        self.act = nn.SiLU()
        self.grn = GRN(dim * 4)
        self.pwconv2 = nn.Conv2d(dim * 4, dim, kernel_size=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = self.dropout(x)
        return x + residual

# ----------------------------
# Global Response Normalization (GRN)
# ----------------------------
class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * x * nx + self.beta + x

# ----------------------------
# ConvNeXt Residual Dense Block (CNRDB)
# ----------------------------
class CNRDB(nn.Module):
    def __init__(self, in_channels, growth_channels=60, num_blocks=4):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.in_channels = in_channels
        channels = in_channels

        for _ in range(num_blocks):
            block = nn.Sequential(
                nn.Linear(channels, growth_channels),
                nn.ReLU(inplace=True),
                ConvNeXtV2Block(growth_channels),
                nn.Linear(growth_channels, growth_channels),
                nn.SiLU()
            )
            self.blocks.append(block)
            channels += growth_channels  # for concatenation

        self.final_linear = nn.Linear(channels, in_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ = x.view(B, C, -1).permute(0, 2, 1)  # (B, HW, C)
        features = [x_]

        for block in self.blocks:
            out = torch.cat(features, dim=-1)
            out = block(out)
            features.append(out)

        fused = torch.cat(features, dim=-1)
        fused = self.final_linear(fused)
        fused = self.dropout(fused)
        fused = fused + x_.clone()  # residual
        fused = fused.permute(0, 2, 1).view(B, C, H, W)
        return fused

# ----------------------------
# Fusion and Restoration Head
# ----------------------------
class FusionRestorationNet(nn.Module):
    def __init__(self, in_channels=60, out_channels=4, mid_channels=64, num_blocks=3):
        super().__init__()
        self.head = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.cnrdb_blocks = nn.Sequential(*[
            CNRDB(mid_channels) for _ in range(num_blocks)
        ])
        self.tail = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.head(x)
        x = self.cnrdb_blocks(x)
        x = self.tail(x)
        return x

