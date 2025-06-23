import torch
import torch.nn as nn

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

# Dense Convolution Block inside RRDB
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

# Residual-in-Residual Dense Block
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
        return x + 0.2 * out

# Simple Mamba-like Block from scratch
class MambaBlock2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj_in = nn.Linear(dim, dim * 2)
        self.activation = nn.GELU()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, N, C)
        x = self.norm(x)
        x = self.proj_in(x)
        x = self.activation(x)

        x = x.permute(0, 2, 1)  # (B, C, N)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)  # (B, N, C)

        x = self.proj_out(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x

# Final HDR Refinement Network with custom Mamba block
class MambaHDRRefineNet(nn.Module):
    def __init__(self, in_channels=60, feat_channels=128, num_blocks=4):
        super().__init__()
        self.head = nn.Conv2d(in_channels, feat_channels, 3, padding=1)

        self.mamba_blocks = nn.Sequential(*[
            nn.Sequential(
                RRDB(feat_channels),
                SEBlock(feat_channels),
                MambaBlock2D(feat_channels)
            ) for _ in range(num_blocks)
        ])

        self.tail = nn.Sequential(
            nn.Conv2d(feat_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, f2):
        residual = self.head(x)
        residual = self.mamba_blocks(residual)
        residual = self.tail(residual)
        out = x + residual + f2
        return self.final(out)

# Example usage
if __name__ == "__main__":
    model = MambaHDRRefineNet()
    input_tensor = torch.randn(1, 60, 256, 256)
    residual_tensor = torch.randn(1, 60, 256, 256)
    output = model(input_tensor, residual_tensor)
    print(output.shape)  # Should be (1, 4, 256, 256)
