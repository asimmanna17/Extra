import torch
import torch.nn.functional as F

def infer_large_image(model, image, tile_size=512, stride=384):
    """
    Perform tiled inference on a large image.

    Args:
        model: PyTorch model (in eval mode).
        image: Tensor of shape (1, C, H, W).
        tile_size: Size of each tile (int).
        stride: Overlap stride (int, < tile_size).

    Returns:
        Reconstructed output of shape (1, C_out, H, W).
    """
    _, C, H, W = image.shape
    device = image.device

    output_acc = torch.zeros((1, model(image[:, :, :tile_size, :tile_size]).shape[1], H, W), device=device)
    weight_acc = torch.zeros((1, 1, H, W), device=device)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1, y2 = y, min(y + tile_size, H)
            x1, x2 = x, min(x + tile_size, W)

            # Crop tile
            tile = image[:, :, y1:y2, x1:x2]

            # Pad to tile size if needed
            pad_bottom = tile_size - tile.shape[2]
            pad_right  = tile_size - tile.shape[3]
            tile = F.pad(tile, (0, pad_right, 0, pad_bottom), mode='reflect')

            with torch.no_grad():
                tile_out = model(tile)  # Inference

            # Remove padding from output
            tile_out = tile_out[:, :, :y2 - y1, :x2 - x1]

            # Add to accumulator
            output_acc[:, :, y1:y2, x1:x2] += tile_out
            weight_acc[:, :, y1:y2, x1:x2] += 1

    # Normalize by weights
    output_acc /= weight_acc
    return output_acc
