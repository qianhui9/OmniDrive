"""Invocation-local reuse of deterministic measurements, never state validity."""
from contextlib import contextmanager
from contextvars import ContextVar

MEASUREMENTS = ContextVar('m1_measurements', default=None)


@contextmanager
def measurement_cache(enabled):
    token = MEASUREMENTS.set(dict(gray={}, transforms={}) if enabled else None)
    try:
        yield
    finally:
        MEASUREMENTS.reset(token)
