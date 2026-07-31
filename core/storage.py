"""
core/storage.py
================
Persistence layer for the KV Cache Introspector.

Three concerns live here, all deliberately backed by plain SQLite (stdlib
`sqlite3`, zero extra dependencies, single-file, trivially inspectable with
any SQLite browser):

1. `EventSink` (+ `SQLiteEventSink` / `JSONLEventSink`) -- the per-token,
   per-layer metric stream written by `core.hooks.KVHookManager`.
2. `ExperienceDB` -- one row per compression experiment, consumed by the
   GEPA adapter (`integrations/gepa_adapter.py`) to derive lessons and
   propose prompt updates.
3. `KnowledgeGraphDB` -- a minimal entity/relationship store satisfying the
   "Ontology" success criterion: every Token, Layer, Head, KVCache,
   Experiment and Plugin becomes an entity, and things like
   "Token 103 ATTENDED_BY Token 108" become relationships.

None of these classes import torch/numpy at module scope, so they're usable
standalone (e.g. from a notebook, from the Streamlit debugger, or from a
GEPA/Autoresearch process that never touches a model directly).
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol


# ---------------------------------------------------------------------------
# Event schema (Subproblem 1: "Data storage schema (SQLite/JSONL)")
# ---------------------------------------------------------------------------

@dataclass
class KVEvent:
    """One row = one (layer, token) observation captured by a hook."""
    run_id: str
    ts: float
    layer_idx: int
    token_pos: int
    token_source: str          # "prompt" | "generated"
    kv_bytes: int
    latency_ms: float
    head_idx: Optional[int] = None
    l2_norm: Optional[float] = None
    attn_entropy: Optional[float] = None
    variance: Optional[float] = None
    extra: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        d = asdict(self)
        d["extra_json"] = json.dumps(d.pop("extra"))
        return d


class EventSink(Protocol):
    def write(self, event: KVEvent) -> None: ...
    def read_all(self, run_id: Optional[str] = None) -> list[dict]: ...
    def close(self) -> None: ...


_EVENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    ts REAL NOT NULL,
    layer_idx INTEGER NOT NULL,
    head_idx INTEGER,
    token_pos INTEGER NOT NULL,
    token_source TEXT NOT NULL,
    kv_bytes INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    l2_norm REAL,
    attn_entropy REAL,
    variance REAL,
    extra_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_token ON events(token_pos);
CREATE INDEX IF NOT EXISTS idx_events_layer ON events(layer_idx);
"""


class SQLiteEventSink:
    """Default sink. One `events` table, append-only, indexed for the
    Timeline/Memory debugger's typical queries (by token, by layer)."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(_EVENT_SCHEMA)
        self._conn.commit()

    def write(self, event: KVEvent) -> None:
        row = event.to_row()
        row.pop("id", None)
        cols = ", ".join(row.keys())
        qs = ", ".join("?" for _ in row)
        self._conn.execute(f"INSERT INTO events ({cols}) VALUES ({qs})", tuple(row.values()))
        self._conn.commit()

    def write_many(self, events: Iterable[KVEvent]) -> None:
        rows = [e.to_row() for e in events]
        if not rows:
            return
        for r in rows:
            r.pop("id", None)
        cols = ", ".join(rows[0].keys())
        qs = ", ".join("?" for _ in rows[0])
        self._conn.executemany(
            f"INSERT INTO events ({cols}) VALUES ({qs})",
            [tuple(r.values()) for r in rows],
        )
        self._conn.commit()

    def read_all(self, run_id: Optional[str] = None) -> list[dict]:
        """Without `run_id`, returns every row ever written to this file --
        which, for a persistent path like `data/events_gemma.db` that
        multiple script invocations share, means rows from EVERY past run,
        not just the current one. Pass `run_id` (each run gets a fresh one
        automatically -- see KVHookManager/CacheTap) to scope to one run.
        This is not a hypothetical: it's why several event-count
        "mismatches" earlier in this project's debugging were partly
        counting stale rows from previous runs. See CHANGELOG.md."""
        if run_id is not None:
            cur = self._conn.execute(
                "SELECT * FROM events WHERE run_id = ? ORDER BY token_pos, layer_idx", (run_id,)
            )
        else:
            cur = self._conn.execute("SELECT * FROM events ORDER BY token_pos, layer_idx")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def distinct_run_ids(self) -> list[str]:
        """All run_ids present in this file -- the fastest way to notice
        "oh, this file has 6 old runs in it" before it bites you."""
        cur = self._conn.execute("SELECT DISTINCT run_id FROM events ORDER BY run_id")
        return [row[0] for row in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()


class JSONLEventSink:
    """Append-only JSONL sink -- convenient for streaming/tailing (`tail -f`)
    or piping into other tools that don't want a SQLite dependency."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, event: KVEvent) -> None:
        self._fh.write(json.dumps(event.to_row()) + "\n")
        self._fh.flush()

    def read_all(self, run_id: Optional[str] = None) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        if run_id is not None:
            rows = [r for r in rows if r.get("run_id") == run_id]
        return rows

    def close(self) -> None:
        self._fh.close()


class InMemoryEventSink:
    """No I/O at all -- used by unit tests and by short-lived experiments."""

    def __init__(self):
        self._events: list[KVEvent] = []

    def write(self, event: KVEvent) -> None:
        self._events.append(event)

    def read_all(self, run_id: Optional[str] = None) -> list[dict]:
        rows = [e.to_row() for e in self._events]
        if run_id is not None:
            rows = [r for r in rows if r.get("run_id") == run_id]
        return rows

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# ExperienceDB (Subproblem 5/6: GEPA lessons, pass/fail thresholds)
# ---------------------------------------------------------------------------

_EXPERIENCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS experience (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    experiment_name TEXT NOT NULL,
    plugin TEXT NOT NULL,
    config_json TEXT,
    compression_ratio REAL,
    ppl_delta REAL,
    latency_ms REAL,
    passed INTEGER,
    lesson TEXT
);
"""


class ExperienceDB:
    """One row per compression experiment. This is the table GEPA reads to
    decide how to mutate `prompts/planner.md` / `reflection.md` -- see
    `integrations/gepa_adapter.py`."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(_EXPERIENCE_SCHEMA)
        self._conn.commit()

    def log(
        self,
        experiment_name: str,
        plugin: str,
        config: dict,
        compression_ratio: float,
        ppl_delta: Optional[float],
        latency_ms: float,
        passed: bool,
        lesson: str,
    ) -> int:
        def _safe_default(o):
            # Plugin kwargs can legitimately include numpy arrays (e.g.
            # head_pruning's attn_weights) -- log a compact shape summary
            # instead of either crashing or dumping a huge blob into SQLite.
            shape = getattr(o, "shape", None)
            if shape is not None:
                return f"<{type(o).__name__} shape={tuple(shape)}>"
            return str(o)

        cur = self._conn.execute(
            """INSERT INTO experience
               (ts, experiment_name, plugin, config_json, compression_ratio,
                ppl_delta, latency_ms, passed, lesson)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                time.time(), experiment_name, plugin, json.dumps(config, default=_safe_default),
                compression_ratio, ppl_delta, latency_ms, int(passed), lesson,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def recent(self, n: int = 20) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM experience ORDER BY id DESC LIMIT ?", (n,)
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# KnowledgeGraphDB (Subproblem 6: "Ontology" criterion)
# ---------------------------------------------------------------------------

_KG_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,      -- Token | Layer | Head | KVCache | Experiment | Plugin
    entity_key TEXT NOT NULL,       -- stable string id, e.g. "token:103"
    properties_json TEXT,
    UNIQUE(entity_type, entity_key)
);
CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    src_key TEXT NOT NULL,
    dst_key TEXT NOT NULL,
    relation_type TEXT NOT NULL,    -- ATTENDED_BY | PRODUCED_BY | COMPRESSED_BY | EVICTED_BY
    weight REAL,
    properties_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_rel_src ON relationships(src_key);
CREATE INDEX IF NOT EXISTS idx_rel_dst ON relationships(dst_key);
"""


class KnowledgeGraphDB:
    """Minimal but real entity/relationship store. Deliberately not a graph
    database -- SQLite with two tables is enough for the scale here, and
    keeps the whole framework to zero extra infra to stand up."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(_KG_SCHEMA)
        self._conn.commit()

    def upsert_entity(self, entity_type: str, entity_key: str, **properties) -> None:
        self._conn.execute(
            """INSERT INTO entities (entity_type, entity_key, properties_json)
               VALUES (?,?,?)
               ON CONFLICT(entity_type, entity_key)
               DO UPDATE SET properties_json=excluded.properties_json""",
            (entity_type, entity_key, json.dumps(properties)),
        )
        self._conn.commit()

    def add_relationship(
        self, src_key: str, dst_key: str, relation_type: str,
        weight: Optional[float] = None, **properties,
    ) -> None:
        self._conn.execute(
            """INSERT INTO relationships (src_key, dst_key, relation_type, weight, properties_json)
               VALUES (?,?,?,?,?)""",
            (src_key, dst_key, relation_type, weight, json.dumps(properties)),
        )
        self._conn.commit()

    def neighbors(self, entity_key: str, relation_type: Optional[str] = None) -> list[dict]:
        q = "SELECT * FROM relationships WHERE src_key = ?"
        params: list[Any] = [entity_key]
        if relation_type:
            q += " AND relation_type = ?"
            params.append(relation_type)
        cur = self._conn.execute(q, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def entity_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]

    def relationship_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# MutationDB -- reproducible KVModifier experiments (core/kv_modifiers.py)
# ---------------------------------------------------------------------------

_MUTATION_SCHEMA = """
CREATE TABLE IF NOT EXISTS mutations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    ts REAL NOT NULL,
    mutation_type TEXT NOT NULL,
    config_json TEXT,
    prompt TEXT,
    baseline_output TEXT,
    mutated_output TEXT,
    output_changed INTEGER,
    latency_ms REAL,
    integrity_passed INTEGER
);
"""


class MutationDB:
    """One row per live KV-intervention experiment -- exactly the record
    TODO.txt asked for: mutation type + target + parameters, generated
    output (both baseline and mutated, so a diff is always one query
    away), latency, and whether `verify_capture_integrity` still held
    after the intervention (a modifier that corrupts the cache in a way
    that breaks the layer's own seq-length accounting is a real failure
    mode worth catching, not just "did the text change")."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(_MUTATION_SCHEMA)
        self._conn.commit()

    def log(
        self, run_id: str, mutation_type: str, config: dict, prompt: str,
        baseline_output: str, mutated_output: str, latency_ms: float,
        integrity_passed: bool,
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO mutations
               (run_id, ts, mutation_type, config_json, prompt, baseline_output,
                mutated_output, output_changed, latency_ms, integrity_passed)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id, time.time(), mutation_type, json.dumps(config, default=str), prompt,
                baseline_output, mutated_output, int(baseline_output != mutated_output),
                latency_ms, int(integrity_passed),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def recent(self, n: int = 20) -> list[dict]:
        cur = self._conn.execute("SELECT * FROM mutations ORDER BY id DESC LIMIT ?", (n,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()


def make_sink(kind: str, path: str | Path) -> EventSink:
    """Factory used by config/models.yaml's `storage.kind: sqlite|jsonl|memory`."""
    if kind == "sqlite":
        return SQLiteEventSink(path)
    if kind == "jsonl":
        return JSONLEventSink(path)
    if kind == "memory":
        return InMemoryEventSink()
    raise ValueError(f"Unknown sink kind: {kind!r}")
