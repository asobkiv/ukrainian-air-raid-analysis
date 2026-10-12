"""Reproducible, point-in-time analysis of recorded alert history."""

import os

# Bound BLAS threads when process workers fit independent regional models.
for _key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(_key, "1")

__version__ = "0.1.0"
