"""
Keep the demo screen clean.

THE PROBLEM THIS SOLVES
  On some installs -- most often Anaconda, which ships MKL as its BLAS --
  scikit-learn's solver emits a wall of warnings during training:

      RuntimeWarning: divide by zero encountered in matmul
      RuntimeWarning: overflow encountered in matmul
      RuntimeWarning: invalid value encountered in matmul
"""

from __future__ import annotations

import os
import warnings

_APPLIED = False


def quiet_blas_fp_warnings() -> bool:
    """Silence spurious BLAS floating-point warnings. Returns True if applied.

    Idempotent and safe to call from every entry script.
    Set MLOPS_DEMO_SHOW_FP_WARNINGS=1 to opt out.
    """
    global _APPLIED

    if os.environ.get("MLOPS_DEMO_SHOW_FP_WARNINGS"):
        return False
    if _APPLIED:
        return True

    import numpy as np

    # numpy's own error state: stop turning these flags into warnings.
    np.seterr(over="ignore", divide="ignore", invalid="ignore")

    # Belt and braces -- some are raised through the warnings module from
    # inside library code rather than numpy's error state. Matched narrowly
    # on the message so unrelated RuntimeWarnings still surface.
    warnings.filterwarnings(
        "ignore",
        message=r".*encountered in matmul.*",
        category=RuntimeWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*encountered in (dot|multiply|reduce).*",
        category=RuntimeWarning,
    )

    _APPLIED = True
    return True
