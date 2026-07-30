"""
harness/budget.py
==================
Subproblem 5 (Meta-Harness `budget.py`) / Subproblem 6 ("enforce
memory/token limits"). Reads limits from config/budgets.yaml and enforces
them *before* an experiment runs (`harness/router.py` calls `check()` up
front), so it prevents an OOM rather than only reporting one after the
fact.
"""
from __future__ import annotations

from dataclasses import dataclass


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class BudgetEnforcer:
    max_kv_bytes: int = 512 * 1024 * 1024       # 512 MB default
    max_tokens_in_flight: int = 100_000
    _bytes_used: int = 0
    _tokens_used: int = 0

    @classmethod
    def from_yaml(cls, path: str) -> "BudgetEnforcer":
        import yaml
        with open(path) as fh:
            cfg = yaml.safe_load(fh) or {}
        return cls(
            max_kv_bytes=int(cfg.get("max_kv_bytes", cls.max_kv_bytes)),
            max_tokens_in_flight=int(cfg.get("max_tokens_in_flight", cls.max_tokens_in_flight)),
        )

    def check(self, additional_bytes: int, additional_tokens: int = 0) -> None:
        """Raises BudgetExceeded without mutating state -- use this for a
        dry-run check before committing to reserve()."""
        if self._bytes_used + additional_bytes > self.max_kv_bytes:
            raise BudgetExceeded(
                f"KV budget exceeded: {self._bytes_used + additional_bytes} > {self.max_kv_bytes} bytes"
            )
        if self._tokens_used + additional_tokens > self.max_tokens_in_flight:
            raise BudgetExceeded(
                f"Token budget exceeded: {self._tokens_used + additional_tokens} > {self.max_tokens_in_flight}"
            )

    def reserve(self, additional_bytes: int, additional_tokens: int = 0) -> None:
        self.check(additional_bytes, additional_tokens)
        self._bytes_used += additional_bytes
        self._tokens_used += additional_tokens

    def release(self, bytes_freed: int, tokens_freed: int = 0) -> None:
        self._bytes_used = max(0, self._bytes_used - bytes_freed)
        self._tokens_used = max(0, self._tokens_used - tokens_freed)

    @property
    def utilization(self) -> dict:
        return {
            "bytes_used": self._bytes_used, "max_kv_bytes": self.max_kv_bytes,
            "bytes_pct": round(100 * self._bytes_used / max(self.max_kv_bytes, 1), 2),
            "tokens_used": self._tokens_used, "max_tokens_in_flight": self.max_tokens_in_flight,
        }
