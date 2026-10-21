"""Lossless compact masks for the JSON M1/M2 boundary; coarse tiles are only a fallback."""
import base64
import binascii
import zlib
import numpy as np


def encode_mask(mask):
    array=np.asarray(mask,dtype=bool)
    if array.ndim!=2: raise ValueError('Expected HxW mask')
    payload=zlib.compress(np.packbits(array.ravel(),bitorder='little').tobytes(),level=1)
    return dict(encoding='packbits_little_zlib_base64',shape=list(array.shape),
                data=base64.b64encode(payload).decode('ascii'))


def decode_mask(record):
    if record['encoding']!='packbits_little_zlib_base64': raise ValueError('Unsupported mask encoding')
    h,w=record['shape']
    if not isinstance(h,int) or not isinstance(w,int) or h<=0 or w<=0: raise ValueError('Invalid shape')
    try:
        payload=zlib.decompress(base64.b64decode(record['data'],validate=True))
    except (zlib.error,binascii.Error) as error:
        raise ValueError('Invalid mask payload') from error
    if len(payload)!=(h*w+7)//8: raise ValueError('Invalid mask length')
    return np.unpackbits(np.frombuffer(payload,dtype=np.uint8),bitorder='little')[:h*w].reshape(h,w).astype(bool)
