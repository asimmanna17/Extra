import torch
import torch.nn.functional as F

def infer_large_image_multi_input(model, x1, x2, x3, tile_size=512, stride=384):
    """
    Tiled inference for model with 3 input tensors and 1 output.

    Args:
        model: function or nn.Module taking 3 tensors and outputting 1 tensor.
        x1, x2, x3: Tensors of shape (1, C, H, W) [e.g., (1,8,4000,3000)]
        tile_size: Tile size (int)
        stride: Step size (int)

    Returns:
        Tensor of shape (1, 4, H, W)
    """
    assert x1.shape == x2.shape == x3.shape, "Input shapes must match"
    B, C, H, W = x1.shape
    device = x1.device

    # Get output channels
    with torch.no_grad():
        dummy_input = torch.zeros((1, C, tile_size, tile_size), device=device)
        C_out = model(dummy_input, dummy_input, dummy_input).shape[1]

    output_acc = torch.zeros((1, C_out, H, W), device=device)
    weight_acc = torch.zeros((1, 1, H, W), device=device)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1, y2 = y, min(y + tile_size, H)
            x1, x2 = x, min(x + tile_size, W)

            # Crop and pad
            def crop_pad(t):
                tile = t[:, :, y1:y2, x1:x2]
                pad_bottom = tile_size - tile.shape[2]
                pad_right  = tile_size - tile.shape[3]
                return F.pad(tile, (0, pad_right, 0, pad_bottom), mode='reflect')

            tile1 = crop_pad(x1)
            tile2 = crop_pad(x2)
            tile3 = crop_pad(x3)

            with torch.no_grad():
                tile_out = model(tile1, tile2, tile3)
                tile_out = tile_out[:, :, :y2 - y1, :x2 - x1]

            output_acc[:, :, y1:y2, x1:x2] += tile_out
            weight_acc[:, :, y1:y2, x1:x2] += 1

    return output_acc / weight_acc
