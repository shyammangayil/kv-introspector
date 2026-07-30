"""
compression/plugins.py
=======================
Subproblem 3: the six compression plugins.

Each is pure numpy so it runs anywhere (no GPU, no bitsandbytes, no CUDA
required) and is directly unit-testable. Production notes on swapping in
real low-level kernels are in each class's docstring -- the *algorithms*
here (quantization grid, SVD truncation, windowing, eviction policies,
importance-based pruning) are the real thing; what's simulated is only the
hardware-specific bit-packing, which numpy can't do natively.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from compression.base import CompressionPlugin, CompressedResult


class FP8Quantization(CompressionPlugin):
    """Simulates FP8 E4M3 (4 exponent bits, 3 mantissa bits) rounding,
    numerically. This is a *numerical* simulation of the quantization grid,
    not a hardware fp8 kernel: on an actual CUDA GPU with compute
    capability >= 8.9 (Ada/Hopper+), swap this for
    `tensor.to(torch.float8_e4m3fn)` or `bitsandbytes` for a real
    1-byte-per-element memory win. The reconstruction-error trade-off this
    plugin reports should closely track true fp8, since it rounds to the
    same exponent/mantissa budget.
    """
    name = "fp8_e4m3"
    EXP_BITS, MANT_BITS, BIAS = 4, 3, 7

    def compress(self, kv, **kwargs) -> CompressedResult:
        arr = kv.astype(np.float32)
        max_val = (2 - 2 ** -self.MANT_BITS) * 2 ** (2 ** self.EXP_BITS - 1 - self.BIAS)
        clipped = np.clip(arr, -max_val, max_val)

        sign = np.sign(clipped)
        abs_val = np.where(clipped == 0, 1e-30, np.abs(clipped))
        exponent = np.floor(np.log2(abs_val))
        scale = 2.0 ** (self.MANT_BITS - exponent)
        quantized = sign * np.round(abs_val * scale) / scale
        quantized = np.where(clipped == 0, 0.0, quantized)

        target_format_bytes = arr.size * 1  # real fp8 = 1 byte/element
        return CompressedResult(
            data=quantized.astype(np.float32), original_shape=arr.shape,
            original_dtype=str(arr.dtype),
            compression_ratio=self._ratio(arr, target_format_bytes), latency_ms=0.0,
            metadata={"exp_bits": self.EXP_BITS, "mant_bits": self.MANT_BITS,
                      "max_abs_err": float(np.max(np.abs(arr - quantized)))},
        )

    def decompress(self, compressed: CompressedResult) -> np.ndarray:
        return compressed.data.astype(np.float32)


class INT4Quantization(CompressionPlugin):
    """Group-wise asymmetric INT4 (16 levels), group size 32 -- the same
    scheme `bitsandbytes.functional.quantize_4bit` implements on GPU; this
    is the reference numpy version so the math is testable without CUDA.
    Real int4 storage packs two 4-bit codes per byte; that's what
    `compression_ratio` counts, plus fp32 scale/zero-point overhead per
    group (also real, and non-negligible at small group sizes).
    """
    name = "int4_groupwise"
    GROUP_SIZE = 32

    def compress(self, kv, group_size: Optional[int] = None, **kwargs) -> CompressedResult:
        gs = group_size or self.GROUP_SIZE
        arr = kv.astype(np.float32)
        flat = arr.reshape(-1)
        pad = (-len(flat)) % gs
        if pad:
            flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
        groups = flat.reshape(-1, gs)

        gmin = groups.min(axis=1, keepdims=True)
        gmax = groups.max(axis=1, keepdims=True)
        scale = np.where((gmax - gmin) == 0, 1e-8, (gmax - gmin) / 15.0)
        q = np.clip(np.round((groups - gmin) / scale), 0, 15).astype(np.uint8)
        recon = q.astype(np.float32) * scale + gmin

        packed_bytes = int(np.ceil(q.size / 2))          # two 4-bit codes/byte
        overhead_bytes = (scale.size + gmin.size) * 4      # fp32 scale + zero per group
        target_format_bytes = packed_bytes + overhead_bytes

        return CompressedResult(
            data={"q": q, "scale": scale, "zero": gmin, "pad": pad, "orig_size": arr.size},
            original_shape=arr.shape, original_dtype=str(arr.dtype),
            compression_ratio=self._ratio(arr, target_format_bytes), latency_ms=0.0,
            metadata={"group_size": gs, "max_abs_err": float(np.max(np.abs(groups - recon))),
                      "overhead_bytes": overhead_bytes},
        )

    def decompress(self, compressed: CompressedResult) -> np.ndarray:
        d = compressed.data
        recon = d["q"].astype(np.float32) * d["scale"] + d["zero"]
        flat = recon.reshape(-1)[: d["orig_size"]]
        return flat.reshape(compressed.original_shape)


class LowRankCompression(CompressionPlugin):
    """Per-head SVD truncation: the [seq_len, head_dim] matrix for each
    head is replaced by its rank-r truncated SVD (r = min(rank, seq_len,
    head_dim)), stored as U_r, S_r, V_r^T. Reconstruction is
    U_r @ diag(S_r) @ V_r^T. Effective mainly once seq_len and head_dim are
    both comfortably larger than r -- on tiny toy shapes the "compression"
    can legitimately come out >= 1x, which is honestly reported, not hidden.
    """
    name = "low_rank_svd"

    def compress(self, kv, rank: int = 16, **kwargs) -> CompressedResult:
        arr = kv.astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        heads, seq_len, head_dim = arr.shape
        r = max(1, min(rank, seq_len, head_dim))

        Us, Ss, Vts = [], [], []
        for h in range(heads):
            u, s, vt = np.linalg.svd(arr[h], full_matrices=False)
            Us.append(u[:, :r]); Ss.append(s[:r]); Vts.append(vt[:r, :])
        U, S, Vt = np.stack(Us), np.stack(Ss), np.stack(Vts)
        recon = np.stack([(U[h] * S[h]) @ Vt[h] for h in range(heads)])

        target_format_bytes = (U.size + S.size + Vt.size) * 4  # fp32
        return CompressedResult(
            data={"U": U, "S": S, "Vt": Vt}, original_shape=arr.shape,
            original_dtype=str(arr.dtype),
            compression_ratio=self._ratio(arr, target_format_bytes), latency_ms=0.0,
            metadata={"rank": r, "max_abs_err": float(np.max(np.abs(arr - recon)))},
        )

    def decompress(self, compressed: CompressedResult) -> np.ndarray:
        d = compressed.data
        heads = d["U"].shape[0]
        recon = np.stack([(d["U"][h] * d["S"][h]) @ d["Vt"][h] for h in range(heads)])
        return recon.reshape(compressed.original_shape)


class SlidingWindow(CompressionPlugin):
    """Keeps only the last `window` tokens, plus (optionally) `sink_tokens`
    fixed tokens from the very start of the sequence -- the StreamingLLM
    "attention sink" trick, which keeps quality much higher than a naive
    window alone because the first few tokens absorb a disproportionate
    share of attention mass in most trained transformers.
    """
    name = "sliding_window"

    def compress(self, kv, window: int = 8, sink_tokens: int = 0, **kwargs) -> CompressedResult:
        arr = kv.astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        seq_len = arr.shape[1]
        keep_idx = sorted(set(
            list(range(min(sink_tokens, seq_len))) +
            list(range(max(0, seq_len - window), seq_len))
        ))
        kept = arr[:, keep_idx, :]

        return CompressedResult(
            data={"kept": kept, "keep_idx": keep_idx, "seq_len": seq_len},
            original_shape=arr.shape, original_dtype=str(arr.dtype),
            compression_ratio=self._ratio(arr, kept.nbytes), latency_ms=0.0,
            metadata={"window": window, "sink_tokens": sink_tokens,
                      "tokens_dropped": seq_len - len(keep_idx), "keep_idx": keep_idx},
        )

    def decompress(self, compressed: CompressedResult) -> np.ndarray:
        # Lossy + shape-changing: dropped tokens are gone, not reconstructable.
        # Returns the [heads, len(keep_idx), head_dim] array of what actually
        # survived -- deliberately NOT zero-padded back to original_shape.
        # An earlier version did zero-pad for shape compatibility, but that
        # made Evaluator.benchmark_plugin's "reconstruction error" actually
        # measure the magnitude of the *discarded* tokens (comparing them to
        # invented zeros) rather than anything about reconstruction quality
        # -- misleading, confirmed by a real run against Gemma 3 where it
        # reported errors of 40-47 that looked like broken math but were
        # really just "these are Gemma's real, large-magnitude KV values."
        # Callers wanting the original positions back should read
        # `compressed.data["keep_idx"]` and place `compressed.data["kept"]`
        # at those indices themselves.
        return compressed.data["kept"]


class TokenEviction(CompressionPlugin):
    """Drops tokens down to `keep_n` under one of three policies:
      - "fifo": drop oldest tokens.
      - "lru": drop tokens least recently accessed (needs `access_order`,
        a list of token indices from least- to most-recently attended).
      - "attention": drop tokens with the lowest importance score
        (`attn_scores`, one score per token; falls back to per-token KV
        norm if not supplied -- a norm-based proxy for "how much this
        token's representation stands out").
    """
    name = "token_eviction"

    def compress(self, kv, keep_n: int = 8, policy: str = "attention",
                 access_order=None, attn_scores=None, **kwargs) -> CompressedResult:
        arr = kv.astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        seq_len = arr.shape[1]
        keep_n = min(keep_n, seq_len)

        if policy == "fifo":
            keep_idx = list(range(seq_len - keep_n, seq_len))
        elif policy == "lru":
            order = list(access_order) if access_order is not None else list(range(seq_len))
            keep_idx = sorted(order[-keep_n:])
        elif policy == "attention":
            scores = np.asarray(attn_scores) if attn_scores is not None else np.linalg.norm(arr, axis=(0, 2))
            keep_idx = sorted(np.argsort(-scores)[:keep_n].tolist())
        else:
            raise ValueError(f"Unknown eviction policy: {policy!r} (use fifo|lru|attention)")

        kept = arr[:, keep_idx, :]
        return CompressedResult(
            data={"kept": kept, "keep_idx": keep_idx, "seq_len": seq_len},
            original_shape=arr.shape, original_dtype=str(arr.dtype),
            compression_ratio=self._ratio(arr, kept.nbytes), latency_ms=0.0,
            metadata={"policy": policy, "keep_n": keep_n, "tokens_dropped": seq_len - keep_n},
        )

    def decompress(self, compressed: CompressedResult) -> np.ndarray:
        # See SlidingWindow.decompress for why this returns the actually-kept
        # shape rather than zero-padding back to original_shape.
        return compressed.data["kept"]


class HeadPruning(CompressionPlugin):
    """Drops (or zeroes) heads whose attention entropy falls below
    `entropy_threshold`: low entropy means a head's attention is peaky /
    near-deterministic, which correlates with redundancy in the classic
    head-pruning literature. Needs `attn_weights` [heads, seq, seq] for the
    entropy signal; falls back to per-head KV-norm as an importance proxy
    if attention weights aren't available. Never prunes every head.
    """
    name = "head_pruning"

    def compress(self, kv, attn_weights=None, entropy_threshold: float = 1.0,
                 drop: bool = True, **kwargs) -> CompressedResult:
        arr = kv.astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        heads = arr.shape[0]

        if attn_weights is not None:
            from core.inspector import KVCacheInspector
            importance = KVCacheInspector.head_importance(attn_weights)
        else:
            importance = np.linalg.norm(arr, axis=(1, 2))

        keep_mask = importance >= entropy_threshold
        if not keep_mask.any():
            keep_mask[int(np.argmax(importance))] = True  # never drop every head
        keep_idx = np.where(keep_mask)[0].tolist()

        if drop:
            kept = arr[keep_idx]
        else:
            kept = arr * keep_mask[:, None, None]  # zeroed in place, same footprint

        return CompressedResult(
            data={"kept": kept, "keep_idx": keep_idx, "num_heads": heads, "dropped": drop},
            original_shape=arr.shape, original_dtype=str(arr.dtype),
            compression_ratio=self._ratio(arr, kept.nbytes), latency_ms=0.0,
            metadata={"heads_pruned": heads - len(keep_idx), "heads_kept": len(keep_idx),
                      "importance": [float(x) for x in importance]},
        )

    def decompress(self, compressed: CompressedResult) -> np.ndarray:
        # See SlidingWindow.decompress for why the drop=True case returns
        # the actually-kept shape rather than zero-padding heads back in.
        d = compressed.data
        return d["kept"]


PLUGIN_REGISTRY: dict[str, type] = {
    "fp8": FP8Quantization,
    "int4": INT4Quantization,
    "low_rank": LowRankCompression,
    "sliding_window": SlidingWindow,
    "token_eviction": TokenEviction,
    "head_pruning": HeadPruning,
}


def get_plugin(name: str) -> CompressionPlugin:
    if name not in PLUGIN_REGISTRY:
        raise KeyError(f"Unknown plugin {name!r}. Available: {list(PLUGIN_REGISTRY)}")
    return PLUGIN_REGISTRY[name]()
