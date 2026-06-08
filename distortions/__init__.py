"""
Distortions: Interactive visualization of distortion in nonlinear embeddings.
"""

from . import geometry
from importlib import import_module


def __getattr__(name):
    if name == "visualization":
        return import_module(".visualization", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "geometry",
    "visualization"
]
