import numpy as np
import cv2
import torch
import pywt

def enrich_input_full(ev0_np):
    """
    ev0_np: input EV0 image, shape (H, W, 3), range [0, 1]
    returns: enriched tensor, shape (31, H, W), dtype float32
    """

    # Ensure proper float32
    ev0_np = np.clip(ev0_np.astype(np.float32), 0.0, 1.0)
    H, W, _ = ev0_np.shape

    # 1. Simulated gamma exposures
    gammas = [0.5, 1.0, 2.0]
    pseudo_exps = [np.power(ev0_np, g) for g in gammas]  # 3x (H, W, 3)
    pseudo_stack = np.concatenate(pseudo_exps, axis=2)  # (H, W, 9)

    # 2. Overexposure mask
    over_mask = (ev0_np > 0.95).astype(np.float32)  # (H, W, 3)

    # 3. Edge map
    gray = cv2.cvtColor(ev0_np, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_map = np.sqrt(sobelx**2 + sobely**2)
    edge_map = np.expand_dims(edge_map, axis=2)  # (H, W, 1)

    # 4. Logarithmic tone mapping
    log_map = np.log1p(10 * ev0_np) / np.log(11.0)  # (H, W, 3)

    # 5. Wavelet decomposition per channel (Haar)
    wavelet_feats = []
    for c in range(3):
        coeffs2 = pywt.dwt2(ev0_np[:, :, c], 'haar')
        LL, (LH, HL, HH) = coeffs2
        wavelet_feats.extend([LL, LH, HL, HH])  # each is (H//2, W//2)
    wavelet_stack = np.stack(wavelet_feats, axis=2)  # shape: (H//2, W//2, 12)
    wavelet_stack = cv2.resize(wavelet_stack, (W, H), interpolation=cv2.INTER_LINEAR)  # upsample to match

    # 6. Inverse CRF (approximate linear space)
    def inv_crf(x):
        return np.power(np.clip(x, 1e-4, 1.0), 2.2)  # gamma 2.2 approximation

    inv_crf_map = inv_crf(ev0_np)  # (H, W, 3)

    # Combine all
    all_feats = np.concatenate([
        pseudo_stack,     # 9
        over_mask,        # 3
        edge_map,         # 1
        log_map,          # 3
        wavelet_stack,    # 12
        inv_crf_map       # 3
    ], axis=2)  # total: 31 channels

    # Convert to torch tensor: (C, H, W)
    enriched_tensor = torch.from_numpy(all_feats).permute(2, 0, 1).float()
    return enriched_tensor
