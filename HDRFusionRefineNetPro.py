import torch
import torch.nn as nn
import torch.nn.functional as F

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
# ConvNeXt V2 Block (depth-wise + GRN)
# ----------------------------
class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, dim * 4, 1)
        self.act = nn.SiLU()
        self.grn = GRN(dim * 4)
        self.pwconv2 = nn.Conv2d(dim * 4, dim, 1)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        res = x
        x = self.dwconv(x)
        x = self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = self.dropout(x)
        return res + x

# ----------------------------
# ConvNeXt Residual Dense Block (CNRDB)
# ----------------------------
class CNRDB(nn.Module):
    def __init__(self, in_channels, growth_channels=64, num_blocks=5):
        super().__init__()
        self.blocks = nn.ModuleList()
        current_channels = in_channels
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(current_channels, growth_channels, 1),
                nn.ReLU(inplace=True),
                ConvNeXtV2Block(growth_channels),
                nn.Conv2d(growth_channels, growth_channels, 1),
                nn.SiLU()
            ))
            current_channels += growth_channels
        self.final = nn.Conv2d(current_channels, in_channels, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        features = [x]
        for block in self.blocks:
            concat = torch.cat(features, dim=1)
            out = block(concat)
            features.append(out)
        out = torch.cat(features, dim=1)
        out = self.final(out)
        out = self.dropout(out)
        return x + out

# ----------------------------
# Squeeze-and-Excitation (SE) + GRN Refinement Block
# ----------------------------
class RefineBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//8, channels, 1),
            nn.Sigmoid()
        )
        self.grn = GRN(channels)
    def forward(self, x):
        res = self.body(x)
        scale = self.se(res)
        res = res * scale
        res = self.grn(res)
        return x + res

# ----------------------------
# Gated Fusion of f2 with attention
# ----------------------------
class GatedFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x, f2):
        fusion = torch.cat([x, f2], dim=1)
        weight = self.gate(fusion)
        return x * (1 - weight) + f2 * weight

# ----------------------------
# Final HDR Fusion + Refinement Network
# ----------------------------
class HDRFusionRefineNetPro(nn.Module):
    def __init__(self, in_channels=60, feat_channels=64, num_fusion_blocks=3, num_refine_blocks=8):
        super().__init__()
        self.enc = nn.Conv2d(in_channels, feat_channels, 3, 1, 1)
        self.fusion_blocks = nn.Sequential(*[
            CNRDB(feat_channels) for _ in range(num_fusion_blocks)
        ])
        self.proj_back = nn.Conv2d(feat_channels, in_channels, 3, 1, 1)

        self.gated_fuse = GatedFusion(in_channels)

        self.refine = nn.Sequential(*[
            RefineBlock(in_channels) for _ in range(num_refine_blocks)
        ])

        self.out = nn.Sequential(
            nn.Conv2d(in_channels, 4, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, f2):
        feat = self.enc(x)
        feat = self.fusion_blocks(feat)
        feat = self.proj_back(feat)

        x = self.gated_fuse(feat, f2)
        x = self.refine(x)

        return self.out(x)
