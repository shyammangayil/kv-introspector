"""
debugger/app_streamlit.py
==========================
Subproblem 4: Timeline Debugger + Memory Debugger.

Run with: `streamlit run debugger/app_streamlit.py` from the repo root,
after `python -m examples.run_demo` has populated `data/events.db` and
`data/demo_output.json` (or point the sidebar at any other run's output --
this reads the same schema `core/hooks.py` writes for a real model, so it
works unmodified once you're hooking real Gemma generations too).

Two tabs:
  - Timeline: horizontal bars for cumulative KV cache size per token (like
    a Chrome DevTools waterfall), a latency-per-token line for spotting
    spikes, and a per-token drill-down table across layers.
  - Memory: a token -> nearest-neighbor dependency graph ("Token 5 is most
    similar to Tokens 8, 2"), built from `core.inspector`'s cosine
    similarity, with the edge weights shown on hover/selection.

Design note: `data/events.db` stores lightweight scalar stats per
(layer, token) -- not full K/V tensors -- since that's what scales to a
real model's cache. The Memory tab's similarity graph therefore reads a
precomputed snapshot (`demo_output.json`, written by run_demo.py from the
one layer it inspected in depth) rather than recomputing from raw tensors
live. A production deployment wanting on-demand graphs for an arbitrary
layer would add a small ring-buffer tensor cache alongside the events
table; out of scope here, called out in README.md's uncertainties.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root importable under `streamlit run`

from core.storage import SQLiteEventSink  # noqa: E402

st.set_page_config(page_title="KV Cache Introspector", layout="wide")
st.title("KV Cache Introspector — Timeline & Memory Debugger")

with st.sidebar:
    st.header("Data source")
    db_path = st.text_input("events.db path", value="data/events.db")
    load = st.button("Load", type="primary")

if "events_df" not in st.session_state or load:
    path = Path(db_path)
    if not path.exists():
        st.warning(f"`{path}` not found — run `python -m examples.run_demo` first, "
                   "or point this at another run's events.db.")
        st.stop()
    sink = SQLiteEventSink(path)
    st.session_state["events_df"] = pd.DataFrame(sink.read_all())
    sink.close()

df = st.session_state["events_df"]
if df.empty:
    st.warning("No events in this database yet.")
    st.stop()

tab_timeline, tab_memory = st.tabs(["Timeline Debugger", "Memory Debugger"])

# --------------------------------------------------------------------- #
# Timeline Debugger
# --------------------------------------------------------------------- #
with tab_timeline:
    st.caption(
        "One bar per token position: cumulative KV cache bytes through that step, "
        "colored by prompt vs. generated. Pick a token below for its per-layer breakdown."
    )

    per_token = (
        df.groupby("token_pos")
        .agg(bytes_this_token=("kv_bytes", "sum"), token_source=("token_source", "first"),
             mean_latency_ms=("latency_ms", "mean"), mean_l2_norm=("l2_norm", "mean"),
             mean_attn_entropy=("attn_entropy", "mean"))
        .reset_index().sort_values("token_pos")
    )
    per_token["cumulative_bytes"] = per_token["bytes_this_token"].cumsum()

    fig = go.Figure()
    colors = per_token["token_source"].map({"prompt": "#2b6cb0", "generated": "#dd6b20"})
    fig.add_bar(
        x=per_token["cumulative_bytes"], y=per_token["token_pos"], orientation="h",
        marker_color=colors,
        hovertext=[
            f"token {r.token_pos} ({r.token_source})<br>cumulative: {r.cumulative_bytes} B<br>"
            f"latency: {r.mean_latency_ms:.3f} ms"
            for r in per_token.itertuples()
        ],
        hoverinfo="text",
    )
    fig.update_layout(
        height=max(320, 24 * len(per_token)), xaxis_title="Cumulative KV cache size (bytes)",
        yaxis_title="Token position", yaxis=dict(autorange="reversed"), showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")

    latency_fig = go.Figure()
    latency_fig.add_scatter(x=per_token["token_pos"], y=per_token["mean_latency_ms"], mode="lines+markers")
    latency_fig.update_layout(
        height=220, xaxis_title="Token position", yaxis_title="Mean hook latency (ms)",
        title="Latency per token (spikes = slower forward passes)",
    )
    st.plotly_chart(latency_fig, width="stretch")

    selected_token = st.selectbox("Inspect token", options=per_token["token_pos"].tolist())
    layer_rows = df[df["token_pos"] == selected_token].sort_values("layer_idx")
    st.subheader(f"Token {selected_token} — per-layer stats")
    st.dataframe(
        layer_rows[["layer_idx", "l2_norm", "attn_entropy", "latency_ms", "kv_bytes"]],
        width="stretch",
    )

# --------------------------------------------------------------------- #
# Memory Debugger
# --------------------------------------------------------------------- #
with tab_memory:
    st.caption(
        "Token dependency graph: which tokens' key vectors are most similar to which "
        "(a proxy for redundant/related context). Built by core.inspector.KVCacheInspector"
        ".nearest_neighbors over one layer's captured keys."
    )

    demo_json_path = Path(db_path).parent / "demo_output.json"
    if not demo_json_path.exists():
        st.warning(f"`{demo_json_path}` not found — run `python -m examples.run_demo` to generate it.")
    else:
        dep = json.loads(demo_json_path.read_text())
        graph = dep["dependency_graph"]
        k = st.slider("Neighbors per token to show", 1, max(1, len(graph[0]["neighbors"])), min(2, len(graph[0]["neighbors"])))

        fig2 = go.Figure()
        for n in graph:
            for nb in n["neighbors"][:k]:
                y_offset = 0.35 if nb["token"] > n["token"] else -0.35
                fig2.add_trace(go.Scatter(
                    x=[n["token"], (n["token"] + nb["token"]) / 2, nb["token"]],
                    y=[0, y_offset, 0], mode="lines", line_shape="spline",
                    line=dict(width=1 + 3 * nb["similarity"], color="rgba(99,179,237,0.65)"),
                    hovertext=f"tok{n['token']} ~ tok{nb['token']}  (cos_sim={nb['similarity']:.3f})",
                    hoverinfo="text", showlegend=False,
                ))
        node_x = [n["token"] for n in graph]
        fig2.add_trace(go.Scatter(
            x=node_x, y=[0] * len(node_x), mode="markers+text",
            text=[f"tok{i}" for i in node_x], textposition="top center",
            marker=dict(size=24, color="#2b6cb0", line=dict(width=1, color="white")), showlegend=False,
        ))
        fig2.update_layout(
            height=380, xaxis_title="Token position", yaxis=dict(visible=False),
            title=f"Dependency graph — layer {dep['inspected_layer']} (from {demo_json_path.name})",
        )
        st.plotly_chart(fig2, width="stretch")

        sel = st.selectbox("Token detail", options=[n["token"] for n in graph])
        node = next(n for n in graph if n["token"] == sel)
        st.write(f"**Token {sel}** is most similar to:")
        for nb in node["neighbors"]:
            st.write(f"- Token {nb['token']}  (cosine similarity {nb['similarity']:.3f})")

        st.divider()
        st.subheader("Compression results for this run")
        plugin_df = pd.DataFrame(dep["plugin_results"]).T
        st.dataframe(plugin_df, width="stretch")
