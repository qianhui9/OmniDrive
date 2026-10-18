"""Dependency-correct, opt-in incremental execution of real EVA blocks."""
from .config import M2Config
from .runtime import IncrementalVisionRuntime

__all__ = ['M2Config', 'IncrementalVisionRuntime']
