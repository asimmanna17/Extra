import torch

def histogram_equalization_torch(batch_bayer_img):
    """
    Perform histogram equalization on each channel of a Bayer BGGR image.

    Args:
        batch_bayer_img: Tensor of shape (B, 4, H, W), float in [0,1] or [0,255]

    Returns:
        Equalized tensor of same shape.
    """
    B, C, H, W = batch_bayer_img.shape
    out = torch.zeros_like(batch_bayer_img)

    for b in range(B):
        for c in range(C):
            img = batch_bayer_img[b, c]
            img_flat = img.flatten()

            # Normalize to [0, 255] and cast to int
            img_min, img_max = img.min(), img.max()
            img_uint8 = ((img - img_min) / (img_max - img_min + 1e-8) * 255).to(torch.uint8)

            # Compute histogram (256 bins)
            hist = torch.histc(img_uint8.float(), bins=256, min=0, max=255)

            # Compute cumulative distribution function (CDF)
            cdf = hist.cumsum(0)
            cdf_min = cdf[cdf > 0][0]
            cdf_normalized = (cdf - cdf_min) * 255 / (cdf[-1] - cdf_min + 1e-8)
            cdf_normalized = cdf_normalized.clamp(0, 255).to(torch.uint8)

            # Map original values through CDF
            equalized_flat = cdf_normalized[img_uint8]
            out[b, c] = equalized_flat.reshape(H, W).float() / 255.0

    return out
