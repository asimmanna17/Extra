import torch
import torch.nn as nn
import torch.fft

# Squeeze-and-Excitation Block
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

# Residual Block with SE
class ResSEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            SEBlock(channels)
        )
    def forward(self, x):
        return x + self.block(x)

# DFT Feature Extractor (magnitude + phase or real + imag)
class FFTFeature(nn.Module):
    def __init__(self, use_real_imag=True):
        super().__init__()
        self.use_real_imag = use_real_imag

    def forward(self, x):
        fft = torch.fft.fft2(x)  # (B, C, H, W), complex64
        if self.use_real_imag:
            feat = torch.cat([fft.real, fft.imag], dim=1)  # 2*C channels
        else:
            magnitude = torch.abs(fft)
            phase = torch.angle(fft)
            feat = torch.cat([magnitude, phase], dim=1)
        return feat

# Modified RawHDRRefineNet for input=60, output=4, and FFT fusion
class RawHDRRefineNetFFT(nn.Module):
    def __init__(self, in_channels=60, feat_channels=64, num_blocks=12):
        super().__init__()
        self.in_channels = in_channels
        self.fft_feature = FFTFeature(use_real_imag=True)  # Output: 2*in_channels = 120

        self.f2_proj = nn.Conv2d(2 * in_channels, feat_channels, kernel_size=1)

        self.head = nn.Conv2d(in_channels, feat_channels, 3, padding=1)

        self.body = nn.Sequential(*[ResSEBlock(feat_channels) for _ in range(num_blocks)])

        self.tail = nn.Conv2d(feat_channels, feat_channels, 3, padding=1)

        self.out_proj = nn.Sequential(
            nn.Conv2d(feat_channels, 4, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, f2):
        # Initial residual branch
        residual = self.head(x)
        residual = self.body(residual)
        residual = self.tail(residual)

        # FFT features from f2
        f2_fft_feat = self.fft_feature(f2)  # Shape: (B, 2*in_channels, H, W)
        f2_mapped = self.f2_proj(f2_fft_feat)  # Shape: (B, feat_channels, H, W)

        # Combine
        fused = residual + f2_mapped

        # Final output
        out = self.out_proj(fused)
        return out
