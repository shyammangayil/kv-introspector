"""
core/inspector.py
==================
Subproblem 2: KV Cache Inspector.

Pure, torch-independent statistics functions over KV tensors. Every
function takes a numpy array (or anything array-like -- torch tensors are
accepted and converted) and returns numbers/arrays, with no I/O and no
side effects. This is deliberate: it's the part of the framework that's
cheapest to get right and to unit test, and every other module (the
Timeline/Memory debugger, the compression evaluator, the GEPA adapter)
consumes its output rather than recomputing statistics itself.

Tensor convention: a "KV block" is shaped [num_heads, seq_len, head_dim].
A bare [seq_len, head_dim] array is also accepted (treated as one head).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "numpy"):
            return x.numpy()
    return np.asarray(x)


def _as_3d(kv) -> np.ndarray:
    arr = _to_numpy(kv)
    return arr[None, ...] if arr.ndim == 2 else arr


@dataclass
class TokenStats:
    token_pos: int
    token_source: str
    l2_norm: float
    variance: float
    age: int
    nearest_neighbor: Optional[int] = None
    nearest_neighbor_sim: Optional[float] = None


class KVCacheInspector:
    """Stateless namespace of stat functions -- every method is a
    `@staticmethod`; the class exists only for a tidy call site
    (`KVCacheInspector.l2_norms(...)`)."""

    @staticmethod
    def l2_norms(kv) -> np.ndarray:
        """Per-token L2 norm, averaged across heads. -> [seq_len]"""
        return np.linalg.norm(_as_3d(kv), axis=-1).mean(axis=0)

    @staticmethod
    def variance(kv) -> np.ndarray:
        """Per-token variance across head_dim, averaged across heads. -> [seq_len]"""
        return _as_3d(kv).var(axis=-1).mean(axis=0)

    @staticmethod
    def cosine_similarity_matrix(kv, head: Optional[int] = None) -> np.ndarray:
        """Pairwise token-vs-token cosine similarity for one head, or
        averaged across heads if `head` is None. -> [seq_len, seq_len]"""
        arr = _as_3d(kv)
        heads = arr[head:head + 1] if head is not None else arr
        sims = []
        for h in range(heads.shape[0]):
            v = heads[h]
            norm = np.linalg.norm(v, axis=-1, keepdims=True)
            norm = np.where(norm == 0, 1e-12, norm)
            unit = v / norm
            sims.append(unit @ unit.T)
        return np.mean(sims, axis=0)

    @staticmethod
    def age(current_step: int, token_positions) -> np.ndarray:
        """Decode-steps elapsed since each token entered the cache."""
        return current_step - np.asarray(token_positions)

    @staticmethod
    def token_source(token_positions, prefill_len: int) -> list[str]:
        return ["prompt" if p < prefill_len else "generated" for p in token_positions]

    @staticmethod
    def nearest_neighbors(kv, k: int = 3, exclude_self: bool = True) -> list[list[tuple[int, float]]]:
        """For every token: its k nearest neighbors by cosine similarity.
        -> one list of (neighbor_idx, similarity) per token, used for both
        the duplicate-detection graph and the Memory Debugger."""
        sim = KVCacheInspector.cosine_similarity_matrix(kv)
        n = sim.shape[0]
        results = []
        for i in range(n):
            row = sim[i].copy()
            if exclude_self:
                row[i] = -np.inf
            top_idx = np.argsort(-row)[:k]
            results.append([(int(j), float(sim[i, j])) for j in top_idx])
        return results

    @staticmethod
    def duplicate_clusters(kv, threshold: float = 0.98) -> list[list[int]]:
        """Union-find clustering of near-duplicate token vectors (cosine
        similarity >= threshold). Flags redundant KV entries that are
        candidates for eviction without changing model outputs much."""
        sim = KVCacheInspector.cosine_similarity_matrix(kv)
        n = sim.shape[0]
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= threshold:
                    union(i, j)

        clusters: dict[int, list[int]] = {}
        for i in range(n):
            clusters.setdefault(find(i), []).append(i)
        return [members for members in clusters.values() if len(members) > 1]

    @staticmethod
    def head_importance(attn_weights) -> np.ndarray:
        """Mean attention entropy per head -- low entropy = peaky = a
        candidate for `compression.plugins.HeadPruning`.
        attn_weights: [num_heads, seq_len, seq_len] -> [num_heads]"""
        arr = _as_3d(attn_weights)
        clipped = np.clip(arr, 1e-12, 1.0)
        ent = -np.sum(clipped * np.log(clipped), axis=-1)   # [heads, seq_len]
        return ent.mean(axis=-1)                              # [heads]

    @staticmethod
    def full_report(kv, token_positions, prefill_len: int, current_step: int) -> list[TokenStats]:
        """Convenience: one TokenStats per token position, combining every
        stat above. This is what feeds `success_criteria.md`'s research
        questions and the debugger's per-token drill-down."""
        norms = KVCacheInspector.l2_norms(kv)
        variances = KVCacheInspector.variance(kv)
        ages = KVCacheInspector.age(current_step, token_positions)
        sources = KVCacheInspector.token_source(token_positions, prefill_len)
        neighbors = KVCacheInspector.nearest_neighbors(kv, k=1)
        out = []
        for i, pos in enumerate(token_positions):
            nb_idx, nb_sim = neighbors[i][0] if neighbors[i] else (None, None)
            out.append(TokenStats(
                token_pos=int(pos), token_source=sources[i],
                l2_norm=float(norms[i]), variance=float(variances[i]),
                age=int(ages[i]), nearest_neighbor=nb_idx, nearest_neighbor_sim=nb_sim,
            ))
        return out
