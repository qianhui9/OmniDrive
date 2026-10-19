"""Invocation-local reuse of deterministic measurements, never state validity."""
from contextlib import contextmanager
from contextvars import ContextVar

MEASUREMENTS = ContextVar('m1_measurements', default=None)


@contextmanager
def measurement_cache(enabled, reuse_images=False):
    token = MEASUREMENTS.set(dict(gray={}, transforms={}, image_values={} if reuse_images else None) if enabled else None)
    try:
        yield
    finally:
        MEASUREMENTS.reset(token)


def gray_measurement(gray, derivative=None):
    """Exact float32 conversion/Sobel reused only in this readonly invocation.

    Keep an owner reference so allocator ID reuse cannot alias a different image.
    No flow, association decision, certificate or cross-frame result is cached.
    """
    import cv2
    import numpy as np
    context=MEASUREMENTS.get()
    values=None if context is None else context['image_values']
    key=(id(gray),derivative)
    if values is not None and key in values and values[key][0] is gray:return values[key][1]
    if derivative is None:value=gray.astype(np.float32)/255.
    else:
        image=gray_measurement(gray)
        value=cv2.Sobel(image,cv2.CV_32F,int(derivative==0),int(derivative==1),ksize=3)/8
    if values is not None:values[key]=(gray,value)
    return value
