import torch
import torch.nn as nn
import torch.nn.functional as F

class BrightnessCorrectionBlock(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3):
        super().__init__()
        self.feat_extractor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1)
        )
        self.kernel_pred = nn.Conv2d(in_channels, kernel_size * kernel_size, 3, padding=1)
        self.offset_pred = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, 3, padding=1)
        
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()  # Optional: Soft attention
        )

    def forward(self, F1, F2, F3):
        def process(x):
            f = self.feat_extractor(x)
            k = self.kernel_pred(f)
            θ = self.offset_pred(f)
            # Here you'd use deformable convolution; replacing with standard conv for now
            x_corr = F.relu(f)  # Placeholder for deformable conv
            attn = self.attn(x_corr)
            return x_corr * attn
        
        return process(F1), process(F2), process(F3)


import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import DeformConv2d  # Requires MMCV installed

class BrightnessCorrectionBlock(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.offset_channels = 2 * kernel_size * kernel_size

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1)
        )

        # Predict offsets θ for deformable conv
        self.offset_pred = nn.Conv2d(in_channels, self.offset_channels, 3, padding=1)

        # Deformable convolution
        self.deform_conv = DeformConv2d(in_channels, in_channels, kernel_size=kernel_size, padding=1)

        # Self-attention (lightweight spatial)
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward_single(self, x):
        feat = self.feature_extractor(x)             # Shared features
        offset = self.offset_pred(feat)              # θ Offsets
        deform_feat = self.deform_conv(feat, offset) # Apply deformable convolution
        attn_map = self.attn(deform_feat)            # Self-attention
        return deform_feat * attn_map                # Apply attention weighting

    def forward(self, F1, F2, F3):
        out1 = self.forward_single(F1)
        out2 = self.forward_single(F2)
        out3 = self.forward_single(F3)
        return out1, out2, out3
