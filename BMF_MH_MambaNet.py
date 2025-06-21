import torch
import torch.nn as nn

# ---- BMFBlock ----
class BMFBlock(nn.Module):
    def __init__(self, dim, expansion=2):
        super().__init__()
        hidden_dim = dim * expansion
        self.proj_in = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.vert_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(5, 1), padding=(2, 0), groups=hidden_dim)
        self.horiz_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 5), padding=(0, 2), groups=hidden_dim)
        self.local_conv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=dim)
        self.gate_v = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.gate_h = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.fuse = nn.Conv2d(hidden_dim * 3, dim, kernel_size=1)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        residual = x
        x_proj = self.proj_in(x)
        v = self.vert_conv(x_proj)
        h = self.horiz_conv(x_proj)
        gv = torch.sigmoid(self.gate_v(h))
        gh = torch.sigmoid(self.gate_h(v))
        v = v * gv
        h = h * gh
        local = self.local_conv(x)
        fused = self.fuse(torch.cat([v, h, local], dim=1))
        return self.norm(fused + residual)

# ---- MH_MambaBlock ----
class MH_MambaBlock(nn.Module):
    def __init__(self, dim, num_heads=4, expansion=2, kernel_size=5):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        hidden_dim = self.head_dim * expansion

        self.in_proj = nn.Conv2d(self.head_dim, hidden_dim, kernel_size=1)
        self.vert_convs = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0), groups=hidden_dim)
            for _ in range(num_heads)
        ])
        self.horiz_convs = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2), groups=hidden_dim)
            for _ in range(num_heads)
        ])
        self.gates_v = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
            for _ in range(num_heads)
        ])
        self.gates_h = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
            for _ in range(num_heads)
        ])
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_heads = x.view(B, self.num_heads, self.head_dim, H, W)
        out_heads = []

        for i in range(self.num_heads):
            xi = x_heads[:, i, :, :, :]
            x_proj = self.in_proj(xi)
            v = self.vert_convs[i](x_proj)
            h = self.horiz_convs[i](x_proj)
            gv = torch.sigmoid(self.gates_v[i](h))
            gh = torch.sigmoid(self.gates_h[i](v))
            v = v * gv
            h = h * gh
            out = v + h
            out_heads.append(out)

        out = torch.cat(out_heads, dim=1)
        out = self.out_proj(out)
        return self.norm(out + x)

# ---- Down/Up Sampling ----
def downsample(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

def upsample(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

# ---- BMF_MH_MambaNet ----
class BMF_MH_MambaNet(nn.Module):
    def __init__(self, in_channels=60, out_channels=4, base_dim=64):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1),
            BMFBlock(base_dim)
        )
        self.down1 = downsample(base_dim, base_dim * 2)
        self.enc2 = nn.Sequential(
            BMFBlock(base_dim * 2)
        )
        self.down2 = downsample(base_dim * 2, base_dim * 4)
        self.enc3 = nn.Sequential(
            BMFBlock(base_dim * 4)
        )

        # Bottleneck with Multi-Head Mamba
        self.bottleneck = MH_MambaBlock(base_dim * 4, num_heads=4)

        # Decoder
        self.up2 = upsample(base_dim * 4, base_dim * 2)
        self.dec2 = BMFBlock(base_dim * 2)
        self.up1 = upsample(base_dim * 2, base_dim)
        self.dec1 = BMFBlock(base_dim)

        # Output
        self.head = nn.Conv2d(base_dim, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        # Bottleneck
        b = self.bottleneck(e3)

        # Decoder
        d2 = self.dec2(self.up2(b) + e2)
        d1 = self.dec1(self.up1(d2) + e1)

        # Output
        return self.head(d1)
