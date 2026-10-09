"""Thread-safe Alpha Vantage call budget: rolling minute + UTC day, plus rate-limit pause.

The budget is in-process and best-effort: a restart resets it, while Alpha
Vantage's server-side quota does not. Detecting rate-limit bodies (``pause``)
is the backstop.

Alpha Vantage also rejects bursts (more than one request per second) with a
rate-limit body. ``acquire`` therefore spaces live calls at least
``min_spacing_seconds`` apart by waiting (``asyncio.sleep``, so the event loop
keeps running) instead of falling back.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime

DEFAULT_PER_MINUTE = 5
DEFAULT_PER_DAY = 25
DEFAULT_PAUSE_SECONDS = 60.0
# A little over Alpha Vantage's 1 request/second, as margin for network jitter.
DEFAULT_MIN_SPACING_SECONDS = 1.1
_WINDOW_SECONDS = 60.0


class CallBudget:
    """Tracks live calls; ``try_acquire``/``acquire`` atomically check and reserve one.

    ``clock`` must return Unix epoch seconds (it also determines the UTC day).
    ``sleep`` is awaited by ``acquire`` to wait for a spaced slot (injectable for tests).
    """

    def __init__(
        self,
        per_minute: int = DEFAULT_PER_MINUTE,
        per_day: int = DEFAULT_PER_DAY,
        pause_seconds: float = DEFAULT_PAUSE_SECONDS,
        clock: Callable[[], float] = time.time,
        min_spacing_seconds: float = DEFAULT_MIN_SPACING_SECONDS,
        sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
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
        self._min_spacing = min_spacing_seconds
        self._sleep = sleep
        self._last_slot: float | None = None

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
            self._record(now)
            return True

    def _record(self, slot: float) -> None:
        self._recent.append(slot)
        self._day_count += 1
        self._last_slot = slot if self._last_slot is None else max(self._last_slot, slot)

    def reserve(self) -> float | None:
        """Reserve one live call at the next spaced slot; return seconds to wait, or None.

        Returns None (nothing reserved) when the minute, day, or pause limits deny
        the call. Otherwise the call is booked at ``max(now, last slot + spacing)``
        and the caller must wait the returned delay before making it. If the wall
        clock stepped backward (the last slot is further ahead than any real queue
        could be), the call is booked one spacing from now instead of waiting out
        the jump.
        """
        with self._lock:
            now = self._clock()
            self._refresh(now)
            if now < self._paused_until:
                return None
            if len(self._recent) >= self._per_minute or self._day_count >= self._per_day:
                return None
            slot = now
            if self._last_slot is not None:
                # At most per_minute calls are booked ahead, so a real queue is never longer.
                if self._last_slot - now > self._per_minute * self._min_spacing:
                    slot = now + self._min_spacing  # the clock stepped backward
                else:
                    slot = max(now, self._last_slot + self._min_spacing)
            self._record(slot)
            return slot - now

    async def acquire(self) -> bool:
        """Reserve one live call, waiting (without blocking the loop) until it is spaced.

        Returns False, without waiting, when the minute, day, or pause limits deny it,
        and False after the wait if a rate-limit ``pause()`` began while it waited.
        """
        delay = self.reserve()
        if delay is None:
            return False
        if delay > 0:
            await self._sleep(delay)
            with self._lock:
                if self._clock() < self._paused_until:
                    return False
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
