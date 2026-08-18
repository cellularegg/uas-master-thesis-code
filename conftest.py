"""Pin an import order that avoids a macOS libomp double-init segfault.

xgboost and torch each bundle their own OpenMP runtime. On Apple Silicon,
importing torch before xgboost in the same process crashes with SIGSEGV the
first time xgboost builds a DMatrix. Importing xgboost here, before pytest
collects any test module, fixes the import order for the whole session
regardless of which test module happens to import torch first.
"""

import xgboost  # noqa: F401
