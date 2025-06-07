import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def compute_pseudo_overexposure_score(ref_frame, hdr_frame, intensity_thresh=0.95, delta=0.1):
    """
    ref_frame: numpy array HWC float32 [0,1] (LDR)
    hdr_frame: numpy array HWC float32 [0,1] (HDR tone-mapped)
    
    Returns:
        score: scalar float, fraction of overexposed pixels
    """
    bright = ref_frame > intensity_thresh
    recovered = np.abs(hdr_frame - ref_frame) > delta
    overexposed = np.logical_and(bright, recovered)
    score = np.sum(overexposed) / np.prod(ref_frame.shape)
    return score
class LDRHDRDataset(Dataset):
    def __init__(self, ldr_paths, hdr_paths, transform=None):
        self.ldr_paths = ldr_paths
        self.hdr_paths = hdr_paths
        self.transform = transform

    def __len__(self):
        return len(self.ldr_paths)

    def __getitem__(self, idx):
        ldr_img = np.array(Image.open(self.ldr_paths[idx])).astype(np.float32) / 255.0
        hdr_img = np.array(Image.open(self.hdr_paths[idx])).astype(np.float32) / 255.0
        
        score = compute_pseudo_overexposure_score(ldr_img, hdr_img)
        
        if self.transform:
            ldr_img = self.transform(ldr_img)  # e.g. ToTensor(), normalization
        
        return ldr_img, torch.tensor(score, dtype=torch.float32)
import torch.nn as nn
import torch.nn.functional as F

class OverexposureScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # H/2 x W/2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # (B, 32, 1, 1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # output in [0,1]
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x
