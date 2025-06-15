import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d

class ModernHDRBlock(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, expansion=4):
        super().__init__()
        # Multi-scale deformable attention (DCDR-UNet inspired)
        self.deform_conv = DeformConv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.offset_net = nn.Sequential(
            nn.Conv2d(embed_dim, 27, 3, padding=1),  # 2*3*3 offsets + 3*3 mask
            nn.LeakyReLU(0.1)
        )
        
        # Cross-channel attention (PubMed 40096329)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, embed_dim//8, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim//8, embed_dim, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention (MSANLnet inspired)
        self.spatial_att = nn.Sequential(
            nn.Conv2d(embed_dim, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Enhanced residual block
        self.res_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim*expansion, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(embed_dim*expansion, embed_dim, 3, padding=1)
        )
        
        # Non-local fusion (CVPRW 2024)
        self.non_local = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim//2, 1),
            nn.Conv2d(embed_dim//2, embed_dim//2, 3, groups=embed_dim//2, padding=1),
            nn.Conv2d(embed_dim//2, embed_dim, 1)
        )

    def forward(self, x):
        # Generate offsets and mask for deformable conv
        offset_mask = self.offset_net(x)
        offsets, mask = torch.split(offset_mask, [18,9], dim=1)
        mask = torch.sigmoid(mask)
        
        # Deformable convolution
        x_deform = self.deform_conv(x, offsets, mask=mask)
        
        # Dual attention mechanism
        channel_att = self.channel_att(x_deform)
        spatial_att = self.spatial_att(x_deform)
        x_att = x_deform * (channel_att + spatial_att)
        
        # Enhanced residual connection
        x_res = self.res_conv(x_att)
        
        # Non-local fusion
        x_nonlocal = self.non_local(x_att)
        
        # Final output
        return x + x_res + x_nonlocal
# Replace existing ContextAwareTransformerBlock with:
self.hdr_blocks = nn.Sequential(
    *[ModernHDRBlock(embed_dim) for _ in range(4)]  # 4 blocks
)

# In forward pass after fusion:
x = self.hdr_blocks(fused_features)  # (B, embed_dim, H, W)


import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba  # Make sure Mamba is installed


class MambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_in = nn.Conv2d(dim, dim, 1)
        self.mamba = Mamba(d_model=dim)
        self.norm = nn.LayerNorm(dim)
        self.proj_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj_in(x)
        x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)  # (B, H*W, C)
        x_out = x_mamba.transpose(1, 2).view(B, C, H, W)
        return self.proj_out(x_out)


class ExposureAwareAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x, exposure_mask):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Flatten exposure mask and create attention bias
        mask = F.interpolate(exposure_mask, size=(H, W), mode='bilinear', align_corners=False)
        mask_flat = mask.flatten(2)  # (B, 1, H*W)
        bias = 1.0 - mask_flat  # lower attention to overexposed (1→0)

        # attention (with exposure masking via additive biasing)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat, attn_mask=None, key_padding_mask=None)
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
        return self.proj(attn_out)


class HybridHDRDecoder(nn.Module):
    def __init__(self, embed_dim, out_channels=4, heads=4):
        super().__init__()
        self.mamba = MambaBlock(embed_dim)
        self.attn = ExposureAwareAttention(embed_dim, heads=heads)
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, out_channels, 3, padding=1)
        )

    def forward(self, x, exposure_mask):
        x = self.mamba(x)
        x = self.attn(x, exposure_mask)
        return self.fuse(x)
        
def get_overexposure_mask_bayer(raw_bayer, threshold=0.95):
    """
    raw_bayer: Tensor of shape (B, 4, H, W) in BGGR order
        C=0: B
        C=1: G1
        C=2: G2
        C=3: R
    """
    B, C, H, W = raw_bayer.shape
    B_chan = raw_bayer[:, 0]
    G1_chan = raw_bayer[:, 1]
    G2_chan = raw_bayer[:, 2]
    R_chan = raw_bayer[:, 3]
    
    # Average both greens
    G_avg = 0.5 * (G1_chan + G2_chan)

    # Compute pseudo-grayscale intensity
    intensity = 0.299 * R_chan + 0.587 * G_avg + 0.114 * B_chan  # shape: (B, H, W)

    # Threshold to get overexposed mask
    mask = (intensity > threshold).float()  # binary mask: (B, H, W)
    return mask.unsqueeze(1)  # (B, 1, H, W)

