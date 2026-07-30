"""
compression/base.py
====================
Subproblem 3: shared plugin contract.

Every plugin compresses/decompresses plain numpy arrays (torch tensors are
accepted too -- converted at the boundary via `to_numpy`), so plugins are
unit-testable with zero model/GPU in the loop, and the same code path is
what `harness/router.py` calls in a real generation loop.

A note on `CompressedResult.compression_ratio`: it is computed from the
*target format's* real on-disk/on-device byte width (1 byte/element for
fp8, 4 bits packed two-per-byte for int4, etc.) -- not from however these
numpy arrays happen to be represented in Python memory while this repo's
demo runs on CPU without dedicated low-precision kernels. Each plugin's
`_ratio` call says explicitly what it's counting.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "numpy"):
            return x.numpy()
    return np.asarray(x)


@dataclass
class CompressedResult:
    """What every plugin's `compress()` / `run()` returns."""
    data: Any                       # compressed payload, plugin-specific
    original_shape: tuple
    original_dtype: str
    compression_ratio: float        # original_bytes / target_format_bytes
    latency_ms: float               # wall-clock cost of compress() itself
    metadata: dict = field(default_factory=dict)


class CompressionPlugin(ABC):
    """Base class for all six plugins in `compression/plugins.py`."""
    name: str = "base"

    @abstractmethod
    def compress(self, kv: np.ndarray, **kwargs) -> CompressedResult:
        """kv: [num_heads, seq_len, head_dim] (or [seq_len, head_dim] for a
        single head). Returns a CompressedResult; `latency_ms` inside it is
        overwritten by `run()` with a real measurement, so implementations
        can leave it as 0.0."""
        ...

    @abstractmethod
    def decompress(self, compressed: CompressedResult) -> np.ndarray:
        """Inverse of compress(). For lossy, shape-changing plugins
        (sliding window, eviction, head pruning with drop=True) this
        reconstructs a same-shape array with zeros at discarded positions
        -- see each plugin's docstring for exactly what's recoverable."""
        ...

    def run(self, kv, **kwargs) -> CompressedResult:
        """Timed entry point -- use this, not compress() directly, so every
        plugin gets a real latency measurement without repeating
        `time.perf_counter()` boilerplate in each one."""
        arr = to_numpy(kv)
        t0 = time.perf_counter()
        result = self.compress(arr, **kwargs)
        result.latency_ms = (time.perf_counter() - t0) * 1000.0
        return result

    @staticmethod
    def _ratio(original: np.ndarray, target_format_bytes: int) -> float:
        return original.nbytes / max(target_format_bytes, 1)
