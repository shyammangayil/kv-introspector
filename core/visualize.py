"""
core/visualize.py
==================
Subproblem 2 (visualization half): heatmaps, PCA/UMAP projections, and
nearest-neighbor graphs over KV tensors and attention weights.

Static rendering via matplotlib (zero extra install, always available,
headless-safe via the "Agg" backend). `debugger/app_streamlit.py` renders
the *interactive* versions of the timeline/graph views with plotly instead;
these functions are for one-off figures (papers, notebooks, `run_demo.py`'s
saved PNGs) rather than the live debugger.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from core.inspector import KVCacheInspector, _to_numpy


def plot_attention_heatmap(attn_weights, title: str = "Attention weights", ax=None) -> Figure:
    """attn_weights: [seq, seq] (single head) or [heads, seq, seq]
    (averaged across heads for display)."""
    arr = _to_numpy(attn_weights)
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(arr, cmap="magma", aspect="auto")
    ax.set_xlabel("Key token position")
    ax.set_ylabel("Query token position")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="attention weight")
    fig.tight_layout()
    return fig


def plot_pca_projection(kv, labels=None, title: str = "Key-vector PCA projection", ax=None) -> Figure:
    """kv: [seq, head_dim] or [heads, seq, head_dim] (averaged across heads).
    Projects token vectors to 2D. Uses sklearn's PCA if available, else a
    manual SVD fallback (mean-centered X = U S V^T, coords = U*S) so this
    never hard-fails on a minimal install."""
    arr = _to_numpy(kv)
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    arr = arr - arr.mean(axis=0, keepdims=True)
    try:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2).fit_transform(arr)
    except ImportError:
        u, s, _vt = np.linalg.svd(arr, full_matrices=False)
        coords = u[:, :2] * s[:2]

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(5, 4.5))
    color = labels if labels is not None else np.arange(len(coords))
    ax.scatter(coords[:, 0], coords[:, 1], c=color, cmap="viridis", s=70,
               edgecolor="white", linewidth=0.6, zorder=3)
    for i, (x, y) in enumerate(coords):
        ax.annotate(str(i), (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_umap_projection(kv, labels=None, title: str = "Key-vector UMAP projection", ax=None) -> Figure:
    """Same contract as plot_pca_projection. `umap-learn` pulls in numba and
    is a heavy, slow install for an alternate projection, so this degrades
    to PCA (with a note in the title) when it isn't present rather than
    failing the caller."""
    try:
        import umap
        arr = _to_numpy(kv)
        if arr.ndim == 3:
            arr = arr.mean(axis=0)
        n_neighbors = max(2, min(15, len(arr) - 1))
        coords = umap.UMAP(n_neighbors=n_neighbors, n_components=2).fit_transform(arr)
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(5, 4.5))
        color = labels if labels is not None else np.arange(len(coords))
        ax.scatter(coords[:, 0], coords[:, 1], c=color, cmap="viridis", s=70)
        ax.set_title(title)
        fig.tight_layout()
        return fig
    except ImportError:
        return plot_pca_projection(kv, labels, title=title + "  [umap-learn not installed -> showing PCA]", ax=ax)


def plot_nearest_neighbor_graph(
    kv, k: int = 2, threshold: float = 0.0,
    title: str = "Nearest-neighbor / duplicate graph", ax=None,
) -> Figure:
    """Directed graph: token i -> its k nearest neighbors by cosine
    similarity (edges below `threshold` dropped). Thicker edge = higher
    similarity. This is the static-figure counterpart of the Memory
    Debugger's live dependency graph."""
    import networkx as nx

    neighbors = KVCacheInspector.nearest_neighbors(kv, k=k)
    g = nx.DiGraph()
    g.add_nodes_from(range(len(neighbors)))
    for i, nbrs in enumerate(neighbors):
        for j, sim in nbrs:
            if sim >= threshold:
                g.add_edge(i, j, weight=sim)

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(5.5, 5))
    pos = nx.spring_layout(g, seed=7)
    weights = [g[u][v]["weight"] for u, v in g.edges()] or [0.0]
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color="#2b6cb0", node_size=380)
    nx.draw_networkx_labels(g, pos, ax=ax, font_color="white", font_size=8)
    nx.draw_networkx_edges(
        g, pos, ax=ax, width=[0.6 + 3 * w for w in weights], alpha=0.55,
        edge_color="#63b3ed", connectionstyle="arc3,rad=0.08", arrows=True,
    )
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def save(fig: Figure, path: str) -> str:
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
