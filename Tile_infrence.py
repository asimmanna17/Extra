import torch
import torch.nn.functional as F

def infer_large_image_total_time(model, x1, x2, x3, tile_size=1024, stride=896):
    """
    Run tiled inference and return both output and total inference time (in seconds).
    """
    assert x1.shape == x2.shape == x3.shape
    B, C, H, W = x1.shape
    device = x1.device

    with torch.no_grad():
        dummy_input = torch.zeros((1, C, tile_size, tile_size), device=device)
        C_out = model(dummy_input, dummy_input, dummy_input).shape[1]

    output_acc = torch.zeros((1, C_out, H, W), device=device)
    weight_acc = torch.zeros((1, 1, H, W), device=device)

    # Timing setup
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1, y2 = y, min(y + tile_size, H)
            x1, x2 = x, min(x + tile_size, W)

            def crop_pad(t):
                tile = t[:, :, y1:y2, x1:x2]
                pad_bottom = tile_size - tile.shape[2]
                pad_right  = tile_size - tile.shape[3]
                return F.pad(tile, (0, pad_right, 0, pad_bottom), mode='reflect')

            tile1 = crop_pad(x1)
            tile2 = crop_pad(x2)
            tile3 = crop_pad(x3)

            tile_out = model(tile1, tile2, tile3)
            tile_out = tile_out[:, :, :y2 - y1, :x2 - x1]

            output_acc[:, :, y1:y2, x1:x2] += tile_out
            weight_acc[:, :, y1:y2, x1:x2] += 1

    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)  # milliseconds
    elapsed_sec = elapsed_ms / 1000.0

    print(f"\n🕒 Total Inference Time for 12MP image: {elapsed_sec:.3f} seconds")

    return output_acc / weight_acc
