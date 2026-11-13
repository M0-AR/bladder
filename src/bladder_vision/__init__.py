"""Bladder Vision: a research-grade cystoscopy AI pipeline.

The package is intentionally light at import time. Computer-vision and machine-learning
dependencies are imported inside the functions that need them, so utilities such as
annotation conversion, dataset splitting and metrics work in a minimal Python environment.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
