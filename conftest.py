"""Pytest bootstrap, loaded before any test module.

macOS Intel: torch, faiss-cpu, and lightgbm each bundle their own OpenMP
runtime (libomp). Multiple copies in one process cause segfaults/aborts
on first heavy native call. Mitigations, in order: tell the runtime to
tolerate duplicates (KMP_DUPLICATE_LIB_OK, the standard macOS workaround
for this exact torch+lightgbm clash), import torch first so its libomp
owns the process, and cap faiss threads. None of this is needed on
Linux/Colab, where each wheel links a shared system runtime.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch  # noqa: E402,F401  (must precede faiss/lightgbm imports)

try:
    import faiss
    faiss.omp_set_num_threads(1)
except ImportError:
    pass
