"""EVA's camera-major window layout; padding is outside the attention mask."""
from functools import lru_cache
import numpy as np
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


@lru_cache(maxsize=8)
def window_geometry(batch, height, width, size):
    """Immutable CPU geometry; no per-frame CUDA nonzero/validity construction."""
    rows,cols=(height+size-1)//size,(width+size-1)//size
    c,wy,wx,dy,dx=np.indices((batch,rows,cols,size,size))
    yy=wy*size+dy;xx=wx*size+dx
    valid=(yy<height)&(xx<width)
    shape=(-1,size,size)
    # Invalid coordinates are never used after normalization/padding masking.
    coordinates=tuple(np.ascontiguousarray(a.reshape(shape)) for a in (c,yy.clip(0,height-1),xx.clip(0,width-1)))
    mask=np.ascontiguousarray(valid.reshape(shape))
    cc,y,x=np.indices((batch,height,width))
    grid=(((cc*rows+y//size)*cols+x//size)*size*size+(y%size)*size+x%size).reshape(batch,height*width)
    for a in (*coordinates,mask,grid):a.flags.writeable=False
    return coordinates,mask,grid
