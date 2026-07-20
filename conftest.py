"""Pytest bootstrap, loaded before any test module.

macOS Intel: faiss-cpu and torch each bundle their own OpenMP runtime.
If faiss loads first, the first heavy torch op can segfault from the
duplicate-runtime clash. Importing torch here forces torch's libomp to
own the process before any test file imports faiss; capping faiss's
threads avoids the remaining contention. Harmless everywhere else.
"""
import torch  # noqa: F401  (must precede any faiss import)

try:
    import faiss
    faiss.omp_set_num_threads(1)
except ImportError:
    pass
