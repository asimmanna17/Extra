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
