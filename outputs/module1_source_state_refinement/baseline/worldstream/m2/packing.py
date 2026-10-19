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


class PackBuffer:
    """One bounded scratch buffer per executor, reused for the current shape."""
    def __init__(self):
        self.tensor = None

    def gather(self, windows, ids):
        shape = (len(ids),) + tuple(windows.shape[1:])
        if (self.tensor is None or tuple(self.tensor.shape) != shape
                or self.tensor.dtype != windows.dtype or self.tensor.device != windows.device):
            self.tensor = torch.empty(shape, device=windows.device, dtype=windows.dtype)
        torch.index_select(windows, 0, ids, out=self.tensor)
        return self.tensor

    def clear(self):
        self.tensor = None


class SpatialWindowGather:
    """Bounded shape metadata; gather selected windows without partitioning all channels."""
    def __init__(self):
        self.key = None

    def clear(self):
        self.__dict__.clear()
        self.key = None

    def gather(self, x, size, ids):
        b,h,w,_ = x.shape
        key = (b,h,w,size,x.device)
        if key != self.key:
            self.clear()
            nh,nw = (h+size-1)//size,(w+size-1)//size
            wid = torch.arange(b*nh*nw,device=x.device)
            off = torch.arange(size,device=x.device)
            self.camera = (wid//(nh*nw))[:,None,None]
            row = ((wid//nw)%nh)[:,None,None]*size+off[None,:,None]
            col = (wid%nw)[:,None,None]*size+off[None,None,:]
            self.valid = ((row<h)&(col<w))[...,None]
            self.row,self.col = row.clamp_max(h-1),col.clamp_max(w-1)
            self.key = key
        valid = self.valid.index_select(0,ids)
        packed = x[self.camera.index_select(0,ids), self.row.index_select(0,ids),
                   self.col.index_select(0,ids)]
        return packed.masked_fill(~valid,0),valid
