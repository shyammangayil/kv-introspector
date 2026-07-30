"""
harness/retry.py
=================
Subproblem 5 (Meta-Harness `retry.py`) / Subproblem 6 (Resilience
criteria: "Retries failed hooks 3x before aborting", "Handles OOM errors
by falling back to full-precision KV"). Generic retry+fallback wrapper used
by `harness/router.py` around every plugin invocation, and reusable around
hook attachment or any other risky call.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, TypeVar

T = TypeVar("T")

# Exception *names* (not imported types) so this module doesn't need to
# import torch just to recognize torch.cuda.OutOfMemoryError.
_OOM_EXCEPTION_NAMES = {"OutOfMemoryError", "CudaOutOfMemoryError", "CudaError"}


def _is_oom_like(exc: BaseException) -> bool:
    return isinstance(exc, MemoryError) or type(exc).__name__ in _OOM_EXCEPTION_NAMES


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    backoff_seconds: float = 0.1
    backoff_multiplier: float = 2.0
    attempts_log: list = field(default_factory=list)

    def run(self, fn: Callable[[], T], fallback: Optional[Callable[[], T]] = None) -> T:
        """Calls `fn()` up to `max_attempts` times with exponential
        backoff. On final failure: calls `fallback()` if provided (e.g.
        Router._fallback_report, which returns a full-precision result and
        marks the experiment failed) instead of propagating, otherwise
        re-raises the last exception."""
        delay = self.backoff_seconds
        last_exc: Optional[BaseException] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = fn()
                self.attempts_log.append({"attempt": attempt, "ok": True})
                return result
            except Exception as exc:  # noqa: BLE001 - intentionally broad: generic retry wrapper
                last_exc = exc
                self.attempts_log.append({
                    "attempt": attempt, "ok": False,
                    "error": str(exc), "oom_like": _is_oom_like(exc),
                })
                if attempt < self.max_attempts:
                    time.sleep(delay)
                    delay *= self.backoff_multiplier
        if fallback is not None:
            return fallback()
        assert last_exc is not None
        raise last_exc
