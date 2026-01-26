# gt_map/config.py
import os
import logging

# Set up basic logging if not already configured
logging.basicConfig(level=logging.INFO)

def set_thread_limit(n_threads=1):
    """Limit BLAS/OpenMP threads to avoid oversubscription."""
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS"
    ]:
        os.environ[key] = str(n_threads)

# Apply safe default on import
set_thread_limit(1)
