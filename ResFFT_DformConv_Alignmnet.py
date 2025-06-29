import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torchvision.ops import DeformConv2d
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class ResFFTCovBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )

    def forward(self, x):
        fft_feat = torch.fft.fft2(x, norm='ortho')
        fft_feat = torch.view_as_real(fft_feat)  # (B, C, H, W, 2)
        fft_feat = fft_feat.permute(0, 1, 4, 2, 3).reshape(x.size(0), -1, x.size(2), x.size(3))  # (B, 2C, H, W)
        res = self.resblock(fft_feat)
        return res

class Crossatten_align_atttrans(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, attn=None):
        super().__init__()
        self.attn = attn(dim, (window_size, window_size), num_heads)
        self.window_size = window_size
        self.shift_size = shift_size

    def forward(self, x, y):
        B, H, W, C = x.shape
        overexposure_mask = torch.where(x > 0.95, 0.0, 1.0).mean(dim=-1, keepdim=True)
        x = x * overexposure_mask
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_mask = torch.roll(overexposure_mask, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_y = y
            shifted_mask = overexposure_mask

        x_windows = self.window_partition(shifted_x).view(-1, self.window_size**2, C)
        y_windows = self.window_partition(shifted_y).view(-1, self.window_size**2, C)
        mask_windows = self.window_partition(shifted_mask).squeeze(-1).view(-1, self.window_size**2)

        attn_windows, att = self.attn(x_windows, y_windows, mask=None, overexposure_mask=mask_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        return x, att

    def window_partition(self, x):
        B, H, W, C = x.shape
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size, self.window_size, C)
        return windows

    def window_reverse(self, windows, H, W):
        B = int(windows.shape[0] / (H * W / self.window_size / self.window_size))
        x = windows.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

class Pyramid(nn.Module):
    def __init__(self, in_channels, n_feats):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, n_feats, 3, stride=1, padding=1), nn.LeakyReLU(0.1, inplace=True)),
            nn.Sequential(
                nn.Conv2d(n_feats, n_feats, 3, stride=2, padding=1), nn.LeakyReLU(0.1, inplace=True)),
            nn.Sequential(
                nn.Conv2d(n_feats, n_feats, 3, stride=2, padding=1), nn.LeakyReLU(0.1, inplace=True))
        ])

    def forward(self, x):
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats

class Pyramid_CrossattAlign_Atttrans(nn.Module):
    def __init__(self, scales, num_feats, window_size, num_heads=1, attn=None):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.scales = scales
        self.feat_conv = nn.ModuleDict()
        self.align = nn.ModuleDict()
        self.attn = attn
        self.window_size = window_size
        self.num_heads = num_heads
        for i in range(self.scales, 0, -1):
            level = f'l{i}'
            self.align[level] = Crossatten_align_atttrans(num_feats, num_heads, window_size, attn=self.attn)
            if i < self.scales:
                self.feat_conv[level] = nn.Conv2d(num_feats * 3, num_feats, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, ref_feats_list, toalign_feats_list, patch_ratio_list):
        upsample_feat = None
        last_att = None
        for i in range(self.scales, 0, -1):
            level = f'l{i}'
            ref_feat = ref_feats_list[i-1].permute(0, 2, 3, 1)
            toalign_feat = toalign_feats_list[i-1].permute(0, 2, 3, 1)
            aligned_feat, att = self.align[level](ref_feat, toalign_feat)
            feat = aligned_feat.permute(0, 3, 1, 2)
            if i < self.scales:
                patch_ratio = patch_ratio_list[i-1]
                atttransfer_feat = self.atttransfer(toalign_feat, last_att, patch_ratio).permute(0, 3, 1, 2)
                feat = self.feat_conv[level](torch.cat([feat, upsample_feat, atttransfer_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)
                upsample_feat = self.upsample(feat)
            last_att = att
        return self.lrelu(feat)

    def atttransfer(self, toalign_feat, att, patch_ratio):
        B, H, W, C = toalign_feat.shape
        window_size = self.window_size * patch_ratio
        x = toalign_feat.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        bn_windows = windows.shape[0]
        windows = windows.view(bn_windows, self.window_size, patch_ratio, self.window_size, patch_ratio, C)
        windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().flatten(3)
        windows = windows.view(bn_windows, self.window_size * self.window_size, -1)
        att = att.mean(dim=1)
        atttransfer_feat = att @ windows
        atttransfer_feat = atttransfer_feat.view(bn_windows, self.window_size, self.window_size, patch_ratio, patch_ratio, C)
        atttransfer_feat = atttransfer_feat.permute(0, 1, 3, 2, 4, 5).contiguous()
        atttransfer_feat = atttransfer_feat.view(bn_windows, window_size, window_size, C)
        atttransfer_feat = atttransfer_feat.view(B, H // window_size, W // window_size, window_size, window_size, C)
        atttransfer_feat = atttransfer_feat.permute(0, 1, 3, 2, 4, 5).contiguous()
        atttransfer_feat = atttransfer_feat.view(B, H, W, C)
        return atttransfer_feat

class MultiCrossAlign_head_atttrans_res1sepalign(nn.Module):
    def __init__(self, in_c=8, num_heads=4, dim_align=64):
        super().__init__()
        self.conv_f1 = nn.Conv2d(in_c, dim_align, 3, 1, 1)
        self.conv_f2 = nn.Conv2d(in_c, dim_align, 3, 1, 1)
        self.conv_f3 = nn.Conv2d(in_c, dim_align, 3, 1, 1)
        self.pyramid1 = Pyramid(dim_align, dim_align)
        self.pyramid2 = Pyramid(dim_align, dim_align)
        self.pyramid3 = Pyramid(dim_align, dim_align)
        self.align1 = Pyramid_CrossattAlign_Atttrans(scales=3, num_feats=dim_align, num_heads=num_heads, window_size=8, attn=Cross_WindowAttention_ReAtt)
        self.align2 = Pyramid_CrossattAlign_Atttrans(scales=3, num_feats=dim_align, num_heads=num_heads, window_size=8, attn=Cross_WindowAttention_ReAtt)
        self.offset_conv1 = nn.Conv2d(dim_align, 18, 3, padding=1)
        self.offset_conv3 = nn.Conv2d(dim_align, 18, 3, padding=1)
        self.deform_conv1 = DeformConv2d(dim_align, dim_align, 3, padding=1)
        self.deform_conv3 = DeformConv2d(dim_align, dim_align, 3, padding=1)
        self.resfft1 = ResFFTCovBlock(dim_align)
        self.resfft3 = ResFFTCovBlock(dim_align)

    def forward(self, x1, x2, x3):
        x1 = self.conv_f1(x1)
        x2 = self.conv_f2(x2)
        x3 = self.conv_f3(x3)
        x1_feats = self.pyramid1(x1)
        x2_feats = self.pyramid2(x2)
        x3_feats = self.pyramid3(x3)
        aligned_x1 = self.align1(x2_feats, x1_feats, patch_ratio_list=[2, 2, 2])
        aligned_x3 = self.align2(x2_feats, x3_feats, patch_ratio_list=[2, 2, 2])
        offset1 = self.offset_conv1(aligned_x1)
        offset3 = self.offset_conv3(aligned_x3)
        refined_x1 = self.deform_conv1(aligned_x1, offset1) + self.resfft1(aligned_x1)
        refined_x3 = self.deform_conv3(aligned_x3, offset3) + self.resfft3(aligned_x3)
        return [refined_x1, x2, refined_x3, x1, x3]

class Cross_WindowAttention_ReAtt(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, y, mask=None, overexposure_mask=None):
        B_, N, C = x.shape
        q = self.q(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(y).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(y).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        if overexposure_mask is not None:
            mask_q = overexposure_mask.unsqueeze(1)
            mask_k = overexposure_mask.unsqueeze(2)
            mask_attn = mask_q * mask_k
            attn = attn * mask_attn.unsqueeze(1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.relative_position_index.size(0),
            self.relative_position_index.size(1),
            -1).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn1 = self.attn_drop(attn)
        x = (attn1 @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn
