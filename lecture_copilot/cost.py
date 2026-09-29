from __future__ import annotations

import threading

_LABELS = {"stt": "STT", "translate": "переклад", "slides": "слайди"}


class CostTracker:
    """Thread-safe accumulator for API spend reported by OpenRouter `usage.cost`.

    Only OpenRouter responses carry a cost field, so Gemini-direct calls are not
    counted here.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total = 0.0
        self._by_kind: dict[str, float] = {}

    def add(self, amount: float | None, kind: str = "other") -> None:
        if not amount or amount <= 0:
            return
        with self._lock:
            self._total += amount
            self._by_kind[kind] = self._by_kind.get(kind, 0.0) + amount

    def reset(self) -> None:
        with self._lock:
            self._total = 0.0
            self._by_kind.clear()

    def total(self) -> float:
        with self._lock:
            return self._total

    def amount(self, kind: str) -> float:
        with self._lock:
            return self._by_kind.get(kind, 0.0)

    def summary(self) -> str:
        with self._lock:
            detail = " · ".join(
                f"{_LABELS.get(k, k)} ${v:.4f}"
                for k, v in self._by_kind.items()
                if v > 0
            )
            base = f"${self._total:.4f}"
            return f"{base} ({detail})" if detail else base


COST = CostTracker()
