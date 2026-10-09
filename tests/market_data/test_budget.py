"""CallBudget: rolling minute, UTC day, rate-limit pause, atomic reservation."""

from __future__ import annotations

import threading
from datetime import UTC, datetime

import pytest

from finance_assistant.market_data.budget import CallBudget

from .conftest import FakeClock


def test_five_per_rolling_minute(clock: FakeClock) -> None:
    budget = CallBudget(clock=clock)
    for _ in range(5):
        assert budget.try_acquire()
        clock.advance(10)  # calls at t=0,10,20,30,40
    assert not budget.try_acquire()  # t=50: five in the last 60 s
    clock.advance(10)  # t=60: the t=0 call has left the window
    assert budget.try_acquire()
    assert not budget.try_acquire()


def test_twenty_five_per_utc_day(clock: FakeClock) -> None:
    clock.set(datetime(2026, 9, 30, 12, 0, tzinfo=UTC).timestamp())
    budget = CallBudget(clock=clock)
    for _ in range(25):
        assert budget.try_acquire()
        clock.advance(61)
    assert budget.remaining_today == 0
    assert not budget.try_acquire()
    clock.advance(3600)
    assert not budget.try_acquire()


def test_day_resets_at_utc_midnight(clock: FakeClock) -> None:
    clock.set(datetime(2026, 9, 30, 23, 0, tzinfo=UTC).timestamp())
    budget = CallBudget(per_day=2, clock=clock)
    assert budget.try_acquire() and budget.try_acquire()
    assert not budget.try_acquire()
    clock.set(datetime(2026, 10, 1, 0, 0, 1, tzinfo=UTC).timestamp())
    assert budget.try_acquire()


def test_pause_blocks_for_sixty_seconds(clock: FakeClock) -> None:
    budget = CallBudget(clock=clock)
    budget.pause()
    assert budget.paused
    assert not budget.try_acquire()
    clock.advance(59.9)
    assert not budget.try_acquire()
    clock.advance(0.1)
    assert not budget.paused
    assert budget.try_acquire()


def test_denied_attempts_do_not_consume_budget(clock: FakeClock) -> None:
    budget = CallBudget(clock=clock)
    budget.pause()
    for _ in range(10):
        budget.try_acquire()
    clock.advance(60)
    assert budget.remaining_today == 25
    assert budget.remaining_this_minute == 5


def test_reservation_is_atomic_across_threads(clock: FakeClock) -> None:
    budget = CallBudget(clock=clock)
    barrier = threading.Barrier(50)
    results: list[bool] = []
    results_lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        ok = budget.try_acquire()
        with results_lock:
            results.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count(True) == 5


# -- 1 s spacing (Alpha Vantage's burst limit) --------------------------------


async def test_acquire_spaces_back_to_back_calls_one_second_apart(clock: FakeClock) -> None:
    budget = CallBudget(clock=clock, sleep=clock.sleep)
    assert await budget.acquire()
    assert clock.sleeps == []  # the first call never waits
    assert await budget.acquire()
    assert clock.sleeps == [pytest.approx(1.1)]  # the second waits until 1.1 s after the first
    clock.advance(0.4)
    assert await budget.acquire()
    assert clock.sleeps == [pytest.approx(1.1), pytest.approx(0.7)]


async def test_acquire_does_not_wait_when_calls_are_already_spaced(clock: FakeClock) -> None:
    budget = CallBudget(clock=clock, sleep=clock.sleep)
    assert await budget.acquire()
    clock.advance(1.5)
    assert await budget.acquire()
    assert clock.sleeps == []


async def test_acquire_books_concurrent_callers_into_distinct_slots(clock: FakeClock) -> None:
    budget = CallBudget(clock=clock)
    delays = [budget.reserve() for _ in range(3)]  # three callers arrive at the same instant
    assert delays == [0.0, pytest.approx(1.1), pytest.approx(2.2)]


async def test_acquire_denied_by_limits_returns_false_without_waiting(clock: FakeClock) -> None:
    budget = CallBudget(clock=clock, sleep=clock.sleep)
    budget.pause()
    assert not await budget.acquire()
    assert clock.sleeps == []
    assert budget.remaining_this_minute == 5  # a denied attempt reserves nothing


async def test_acquire_keeps_the_minute_limit(clock: FakeClock) -> None:
    budget = CallBudget(clock=clock, sleep=clock.sleep)
    for _ in range(5):
        assert await budget.acquire()
    assert not await budget.acquire()  # five in the last 60 s, spacing or not
    assert clock.sleeps == [pytest.approx(1.1)] * 4


async def test_acquire_waits_without_blocking_the_event_loop() -> None:
    import asyncio
    import time
    from contextlib import suppress

    budget = CallBudget(min_spacing_seconds=0.2)  # real clock, real asyncio.sleep
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.01)

    task = asyncio.create_task(ticker())
    await asyncio.sleep(0)  # let the ticker start
    ticks = 0
    start = time.monotonic()
    assert await budget.acquire() and await budget.acquire()
    elapsed = time.monotonic() - start
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    assert elapsed >= 0.1
    assert ticks >= 1  # other tasks kept running during the wait


async def test_pause_during_the_wait_cancels_the_call(clock: FakeClock) -> None:
    budget = CallBudget(clock=clock)

    async def sleep_then_rate_limited(seconds: float) -> None:
        budget.pause()  # e.g. the previous call got a rate-limit body meanwhile
        clock.advance(seconds)

    budget._sleep = sleep_then_rate_limited  # type: ignore[method-assign]
    assert await budget.acquire()
    assert not await budget.acquire()


def test_backward_clock_step_never_waits_more_than_one_spacing(clock: FakeClock) -> None:
    budget = CallBudget(clock=clock)
    assert budget.reserve() == 0.0
    clock.advance(-3600)  # the wall clock steps back an hour
    assert budget.reserve() == pytest.approx(1.1)
