"""
Runtime configuration utilities for Apple Silicon and cross-platform BLAS setup.
"""
from __future__ import annotations

import io
import os
import platform
import sys
from typing import Tuple

import numpy as np


def configure_runtime() -> None:
    """Configure BLAS backend for optimal Apple Silicon performance."""
    if platform.system() == "Darwin" and "arm" in platform.machine():
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
    else:
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())
        os.environ["MKL_NUM_THREADS"] = "1"


def detect_accelerate() -> Tuple[bool, str]:
    """
    Multi-method detection of the Accelerate framework.

    Returns
    -------
    Tuple[bool, str]
        Flag indicating Accelerate usage and a short status message.
    """
    config = np.__config__

    def _has_accelerate(info) -> bool:
        libs = info.get("libraries", [])
        return any("accelerate" in str(lib).lower() for lib in libs)

    if hasattr(config, "blas_opt_info") and _has_accelerate(config.blas_opt_info):
        return True, "✅ Using Accelerate framework"

    if hasattr(config, "lapack_opt_info") and _has_accelerate(config.lapack_opt_info):
        return True, "✅ Using Accelerate framework"

    try:
        import inspect

        if "mode" in inspect.signature(config.show).parameters:  # type: ignore[attr-defined]
            info = config.show(mode="dicts")  # type: ignore[attr-defined]
            blas_name = info.get("Build Dependencies", {}).get("blas", {}).get("name", "")
            if blas_name == "accelerate":
                return True, "✅ Using Accelerate framework"
    except Exception:
        pass

    capture = io.StringIO()
    try:
        old_stdout = sys.stdout
        sys.stdout = capture
        config.show()  # type: ignore[attr-defined]
    finally:
        sys.stdout = old_stdout

    if "accelerate" in capture.getvalue().lower():
        return True, "✅ Using Accelerate framework"

    return False, "❌ NOT using Accelerate - will be SLOW"
