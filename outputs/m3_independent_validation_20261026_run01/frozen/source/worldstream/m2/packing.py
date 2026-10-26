"""EVA's camera-major window layout; padding is outside the attention mask."""
import torch
import torch.nn.functional as F


def partition(x, size):
    b, h, w, c = x.shape
    hp, wp = (h + size - 1) // size * size, (w + size - 1) // size * size
    x = F.pad(x, (0, 0, 0, wp-w, 0, hp-h))
    return x.reshape(b, hp//size, size, wp//size, size, c).permute(
        0, 1, 3, 2, 4, 5).reshape(-1, size, size, c), (hp, wp)


def unpartition(windows, size, padded, hw):
    hp, wp = padded
    h, w = hw
    b = windows.shape[0] // ((hp//size)*(wp//size))
    return windows.reshape(b, hp//size, wp//size, size, size, -1).permute(
        0, 1, 3, 2, 4, 5).reshape(b, hp, wp, -1)[:, :h, :w].contiguous()


def window_mask(tokens, size):
    windows, _ = partition(tokens[..., None], size)
    return windows.flatten(1).any(1)
