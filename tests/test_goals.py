"""Goal planning math: hand-checked values, edge cases, validation bounds, date -> months."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from finance_assistant.goals import (
    MAX_MONTHS,
    GoalInputError,
    end_of_month,
    monthly_savings,
    months_until,
    validate_goal_inputs,
)


def test_cap5_reference_value_50k_60_months_5_percent() -> None:
    # PMT = 50,000 × (0.05/12) / ((1 + 0.05/12)^60 − 1) = 735.2283… -> 735.23
    plan = monthly_savings(50_000, 60, 5)
    assert plan.monthly_amount == Decimal("735.23")
    assert plan.months == 60
    assert round(plan.monthly_rate, 10) == Decimal("0.0041666667")
    assert round(plan.growth_factor, 6) == Decimal("1.283359")
    assert plan.total_deposited == Decimal("44113.80")
    assert plan.ending_balance == Decimal("50000.11")  # rounded PMT reaches the target within cents
    assert plan.interest_earned == Decimal("5886.31")
    assert plan.interest_earned == plan.ending_balance - plan.current - plan.total_deposited
    assert not plan.already_covered


@pytest.mark.parametrize(
    ("target", "months", "rate", "current", "expected"),
    [
        (30_000, 48, 4, 0, "577.38"),  # 577.3716… rounded up
        (30_000, 48, 7, 0, "543.39"),  # 543.3873…
        (50_000, 60, 5, 10_000, "546.52"),  # (50,000 − 10,000 × 1.283359) × r / (g − 1) = 546.516…
        (1_000, 12, 12, 0, "78.85"),  # 1,000 × 0.01 / (1.01^12 − 1) = 78.8488
    ],
)
def test_hand_checked_values(target, months, rate, current, expected) -> None:
    assert monthly_savings(target, months, rate, current).monthly_amount == Decimal(expected)


def test_current_savings_grow_and_reduce_the_monthly_amount() -> None:
    without = monthly_savings(50_000, 60, 5)
    with_pv = monthly_savings(50_000, 60, 5, current=10_000)
    assert with_pv.current_future_value == Decimal("12833.59")
    assert with_pv.monthly_amount < without.monthly_amount
    assert with_pv.interest_earned == with_pv.ending_balance - Decimal(10_000) - with_pv.total_deposited


def test_zero_rate_is_remaining_amount_over_months() -> None:
    plan = monthly_savings(50_000, 60, 0)
    assert plan.monthly_amount == Decimal("833.34")  # 833.333… rounded up, never short of the target
    assert plan.growth_factor == 1
    assert plan.interest_earned == 0
    assert plan.ending_balance == Decimal("50000.40")
    assert monthly_savings(12_000, 12, 0, current=6_000).monthly_amount == Decimal("500.00")


def test_already_covered_needs_no_monthly_saving() -> None:
    plan = monthly_savings(50_000, 60, 5, current=40_000)
    assert plan.already_covered
    assert plan.monthly_amount == 0 and plan.total_deposited == 0
    assert plan.current_future_value == Decimal("51334.35")
    assert plan.ending_balance == Decimal("51334.35")
    assert plan.interest_earned == Decimal("11334.35")
    zero_rate = monthly_savings(10_000, 12, 0, current=10_000)
    assert zero_rate.already_covered and zero_rate.interest_earned == 0


def test_rounded_monthly_amount_never_falls_short() -> None:
    for args in [(50_000, 60, 0), (30_000, 48, 4), (30_000, 48, 7), (50_000, 60, 5, 10_000), (1_000, 7, 3)]:
        plan = monthly_savings(*args)
        assert plan.ending_balance >= plan.target, args


def test_sub_cent_payment_rounds_up_and_is_not_already_covered() -> None:
    plan = monthly_savings(500, 1200, 7)  # exact PMT is far below half a cent
    assert plan.monthly_amount == Decimal("0.01")
    assert not plan.already_covered
    assert plan.ending_balance > plan.target


def test_very_large_valid_inputs_do_not_overflow_the_context() -> None:
    covered = monthly_savings(Decimal("1e27"), MAX_MONTHS, 50, current=10_000_000)
    assert covered.already_covered and covered.monthly_amount == 0  # PV grows past the target
    assert covered.ending_balance == covered.current_future_value > covered.target
    needed = monthly_savings(Decimal("1e27"), MAX_MONTHS, 1, current=10_000_000)
    assert not needed.already_covered and needed.monthly_amount > 0
    assert needed.ending_balance >= needed.target


def test_accepts_floats_strings_and_decimals_alike() -> None:
    a = monthly_savings(50000.0, 60, 5.0)
    b = monthly_savings("50000", 60, "5")
    c = monthly_savings(Decimal("50000"), 60, Decimal("5"))
    assert a.monthly_amount == b.monthly_amount == c.monthly_amount == Decimal("735.23")
    assert monthly_savings(50_000, 60, 4.5).monthly_amount == monthly_savings(50_000, 60, "4.5").monthly_amount


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"target": 0}, "target"),
        ({"target": -100}, "target"),
        ({"months": 0}, "months"),
        ({"months": -3}, "months"),
        ({"months": MAX_MONTHS + 1}, "months"),
        ({"annual_rate": -0.1}, "annual_rate"),
        ({"annual_rate": 50.01}, "annual_rate"),
        ({"current": -1}, "current"),
    ],
)
def test_validation_bounds(kwargs: dict, field: str) -> None:
    args = {"target": 10_000, "months": 12, "annual_rate": 5, "current": 0, **kwargs}
    problems = validate_goal_inputs(**args)
    assert [p.field for p in problems] == [field]
    with pytest.raises(GoalInputError) as excinfo:
        monthly_savings(**args)
    assert excinfo.value.problems == problems


def test_validation_edges_are_inclusive() -> None:
    assert validate_goal_inputs(0.01, 1, 0, 0) == []
    assert validate_goal_inputs(1, MAX_MONTHS, 50, 0) == []
    plan = monthly_savings(1_000_000, MAX_MONTHS, 50)  # computes without overflow
    assert plan.months == MAX_MONTHS and plan.monthly_amount == Decimal("0.01")
    assert not plan.already_covered and plan.ending_balance >= plan.target


def test_validation_reports_every_problem_in_order() -> None:
    problems = validate_goal_inputs(-1, 0, 60, -5)
    assert [p.field for p in problems] == ["target", "months", "annual_rate", "current"]


def test_invalid_types_raise() -> None:
    with pytest.raises(TypeError):
        monthly_savings(1_000, 12.5, 5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        monthly_savings("abc", 12, 5)
    with pytest.raises(ValueError):
        monthly_savings(float("inf"), 12, 5)


@pytest.mark.parametrize(
    ("target", "today", "expected"),
    [
        (end_of_month(2030, 6), date(2026, 10, 9), 44),  # "by June 2030"
        (end_of_month(2030, 6), date(2026, 10, 31), 44),  # end-of-month target counts the month
        (date(2030, 6, 8), date(2026, 10, 9), 43),  # one day short of a whole month
        (date(2030, 6, 9), date(2026, 10, 9), 44),
        (date(2026, 11, 9), date(2026, 10, 9), 1),
        (date(2026, 11, 8), date(2026, 10, 9), 0),
        (date(2025, 1, 1), date(2026, 10, 9), -22),
    ],
)
def test_months_until(target: date, today: date, expected: int) -> None:
    assert months_until(target, today) == expected


def test_end_of_month_handles_leap_years() -> None:
    assert end_of_month(2028, 2) == date(2028, 2, 29)
    assert end_of_month(2027, 2) == date(2027, 2, 28)
