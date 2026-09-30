"""CallBudget: rolling minute, UTC day, rate-limit pause, atomic reservation."""

from __future__ import annotations

import threading
from datetime import UTC, datetime

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
