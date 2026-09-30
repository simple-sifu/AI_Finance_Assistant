"""Thread-safe Alpha Vantage call budget: rolling minute + UTC day, plus rate-limit pause.

The budget is in-process and best-effort: a restart resets it, while Alpha
Vantage's server-side quota does not. Detecting rate-limit bodies (``pause``)
is the backstop.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, date, datetime

DEFAULT_PER_MINUTE = 5
DEFAULT_PER_DAY = 25
DEFAULT_PAUSE_SECONDS = 60.0
_WINDOW_SECONDS = 60.0


class CallBudget:
    """Tracks live calls; ``try_acquire`` atomically checks and reserves one.

    ``clock`` must return Unix epoch seconds (it also determines the UTC day).
    """

    def __init__(
        self,
        per_minute: int = DEFAULT_PER_MINUTE,
        per_day: int = DEFAULT_PER_DAY,
        pause_seconds: float = DEFAULT_PAUSE_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._per_minute = per_minute
        self._per_day = per_day
        self._pause_seconds = pause_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._recent: deque[float] = deque()
        self._day: date | None = None
        self._day_count = 0
        self._paused_until = 0.0

    def _utc_day(self, now: float) -> date:
        return datetime.fromtimestamp(now, UTC).date()

    def _refresh(self, now: float) -> None:
        while self._recent and now - self._recent[0] >= _WINDOW_SECONDS:
            self._recent.popleft()
        today = self._utc_day(now)
        if today != self._day:
            self._day = today
            self._day_count = 0

    def try_acquire(self) -> bool:
        """Reserve one live call if the minute, day, and pause limits allow it."""
        with self._lock:
            now = self._clock()
            self._refresh(now)
            if now < self._paused_until:
                return False
            if len(self._recent) >= self._per_minute or self._day_count >= self._per_day:
                return False
            self._recent.append(now)
            self._day_count += 1
            return True

    def pause(self, seconds: float | None = None) -> None:
        """Block live calls for ``seconds`` (default 60 s), e.g. after a rate-limit body."""
        duration = self._pause_seconds if seconds is None else seconds
        with self._lock:
            self._paused_until = max(self._paused_until, self._clock() + duration)

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._clock() < self._paused_until

    @property
    def remaining_today(self) -> int:
        with self._lock:
            self._refresh(self._clock())
            return max(0, self._per_day - self._day_count)

    @property
    def remaining_this_minute(self) -> int:
        with self._lock:
            self._refresh(self._clock())
            return max(0, self._per_minute - len(self._recent))
