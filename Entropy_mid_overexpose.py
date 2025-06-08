import numpy as np
from skimage.color import rgb2gray
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte

def overexposure_score_entropy_only(ref_frame, intensity_thresh=0.9, entropy_thresh=0.4, kernel_size=5):
    """
    Computes overexposure score based only on intensity and entropy from a single LDR reference frame.

    Args:
        ref_frame: np.array (H, W, 3), RGB, float in [0,1]
        intensity_thresh: float, intensity cutoff for bright region
        entropy_thresh: float, entropy cutoff for flat region
        kernel_size: int, size of entropy kernel

    Returns:
        score: float, fraction of overexposed pixels
        overexposed_mask: boolean mask of overexposed regions
        entropy_map: normalized entropy map
    """
    # Step 1: Convert to grayscale
    gray = rgb2gray(ref_frame)

    # Step 2: Convert to uint8 and compute entropy
    gray_uint8 = img_as_ubyte(gray)
    ent = entropy(gray_uint8, disk(kernel_size))
    ent = ent / np.max(ent)  # Normalize to [0,1]

    # Step 3: Create masks
    bright_mask = gray > intensity_thresh
    low_entropy_mask = ent < entropy_thresh

    # Step 4: Combine masks
    overexposed_mask = bright_mask & low_entropy_mask

    # Step 5: Score = proportion of overexposed pixels
    score = np.sum(overexposed_mask) / overexposed_mask.size

    return score, overexposed_mask, ent
import numpy as np
from skimage.color import rgb2gray
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte

def overexposure_score_entropy_only(ref_frame, intensity_thresh=0.9, entropy_thresh=0.4, kernel_size=5):
    gray = rgb2gray(ref_frame)
    gray_uint8 = img_as_ubyte(gray)
    ent = entropy(gray_uint8, disk(kernel_size))
    ent = ent / np.max(ent)
    bright_mask = gray > intensity_thresh
    low_entropy_mask = ent < entropy_thresh
    overexposed_mask = bright_mask & low_entropy_mask
    score = np.sum(overexposed_mask) / overexposed_mask.size
    return score, overexposed_mask, ent

def compare_exposure_entropy_ratio(sht_img, mid_img, entropy_thresh=2.0, radius=5, eps=1e-6):
    def low_entropy_ratio(img): return np.sum(entropy(img_as_ubyte(rgb2gray(img)), disk(radius)) < entropy_thresh) / img.shape[0] / img.shape[1]
    low_entropy_sht = low_entropy_ratio(sht_img)
    raw_ratio = low_entropy_mid / (low_entropy_sht + eps)
    ratio_score = np.log1p(raw_ratio)  # Option A: log-scale for stability
    return ratio_score, low_entropy_mid, low_entropy_sht

def final_overexposure_score(mid_frame, sht_frame, entropy_thresh_mask=0.4, intensity_thresh=0.9, kernel_size_mask=5, entropy_thresh_ratio=2.0, radius_ratio=7, weight_mask=0.6, weight_ratio=0.4):
    mask_score, overexposed_mask, entropy_map = overexposure_score_entropy_only(mid_frame, intensity_thresh, entropy_thresh_mask, kernel_size_mask)
    ratio_score, low_mid, low_sht = compare_exposure_entropy_ratio(sht_frame, mid_frame, entropy_thresh=entropy_thresh_ratio, radius=radius_ratio)
    final_score = weight_mask * mask_score + weight_ratio * ratio_score
    print(f"[INFO] Overexposed mask score: {mask_score:.4f}")
    print(f"[INFO] Entropy ratio (mid/sht): {ratio_score:.4f} (LowE mid={low_mid:.4f}, sht={low_sht:.4f})")
    print(f"[INFO] Final overexposure score: {final_score:.4f}")
    return final_score, overexposed_mask, entropy_map
