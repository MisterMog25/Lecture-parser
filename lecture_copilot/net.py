from __future__ import annotations

import random
import time
from typing import Any, Callable, TypeVar

import httpx

# Transient upstream conditions worth retrying (503 = "model overloaded" etc.)
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

T = TypeVar("T")


def _status_of(exc: Exception) -> int | None:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int):
        return value
    return None


def is_retryable(exc: Exception) -> bool:
    status = _status_of(exc)
    if status is not None:
        return status in RETRY_STATUS
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "503",
            "429",
            "500",
            "502",
            "504",
            "unavailable",
            "overloaded",
            "deadline",
            "timeout",
            "resource exhausted",
            "rate limit",
            "temporarily",
        )
    )


def retry_call(
    fn: Callable[[], T],
    *,
    attempts: int = 5,
    base_delay: float = 0.8,
    max_delay: float = 12.0,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> T:
    """Call fn, retrying transient failures with exponential backoff + jitter."""
    delay = base_delay
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we re-raise unless retryable
            if attempt >= attempts or not is_retryable(exc):
                raise
            last = exc
            if on_retry is not None:
                on_retry(attempt, exc)
            time.sleep(min(max_delay, delay) * (0.5 + random.random() * 0.5))
            delay *= 2
    assert last is not None
    raise last


def request_with_retry(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json: dict[str, Any] | None = None,
    timeout: float = 60.0,
    attempts: int = 5,
    base_delay: float = 0.8,
    max_delay: float = 12.0,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> httpx.Response:
    """HTTP request that retries 429/5xx responses and transport errors."""

    def _do() -> httpx.Response:
        resp = client.request(method, url, headers=headers, json=json, timeout=timeout)
        if resp.status_code in RETRY_STATUS:
            raise httpx.HTTPStatusError(
                f"HTTP {resp.status_code} from {url}",
                request=resp.request,
                response=resp,
            )
        resp.raise_for_status()
        return resp

    return retry_call(
        _do,
        attempts=attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        on_retry=on_retry,
    )
