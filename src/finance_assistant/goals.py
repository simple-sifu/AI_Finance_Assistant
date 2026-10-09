"""Goal planning math (CAP-5): the monthly saving needed to reach a target by a date.

Pure and deterministic, with no LLM imports, so the chat agent and the UI's Goals
tab share one implementation.

Model: monthly compounding at ``annual_rate / 12``, deposits at the end of each
month::

    FV = PV·(1+r)^n + PMT·((1+r)^n − 1)/r

solved for ``PMT``; with ``r = 0``, ``PMT = (FV − PV)/n``. No inflation, taxes,
fees, irregular deposits, or withdrawals.

``annual_rate`` is a percentage (``5`` means 5 % a year). Money is handled as
``Decimal``. The monthly amount is rounded up to the next cent, so the plan never
falls short; other amounts are rounded to cents half-up. Numbers may be passed as ``int``,
``float``, ``str`` or ``Decimal``.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal, localcontext

MAX_MONTHS = 100 * 12
MIN_RATE_PERCENT = Decimal(0)
MAX_RATE_PERCENT = Decimal(50)

_CENT = Decimal("0.01")
_PRECISION = 40

Number = int | float | str | Decimal


@dataclass(frozen=True)
class GoalInputProblem:
    """One goal input that is out of range: which one (``field``) and why (``message``)."""

    field: str  # "target", "months", "annual_rate" or "current"
    message: str


class GoalInputError(ValueError):
    """Raised by ``monthly_savings`` when one or more inputs are out of range."""

    def __init__(self, problems: list[GoalInputProblem]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(p.message for p in self.problems))


@dataclass(frozen=True)
class SavingsPlan:
    """The result of ``monthly_savings``.

    ``monthly_amount`` is rounded up to the next cent, so the plan never falls
    short. ``ending_balance`` is what the plan actually reaches with that rounded
    amount, so it can be above ``target`` (by a few cents from the rounding, or
    more when current savings alone already cover it).
    """

    target: Decimal
    months: int
    annual_rate: Decimal  # percent, e.g. 5 for 5 %
    current: Decimal
    monthly_rate: Decimal  # r = annual_rate / 100 / 12, as a fraction
    growth_factor: Decimal  # (1 + r)^n
    current_future_value: Decimal  # PV·(1+r)^n, rounded to cents
    monthly_amount: Decimal  # PMT, rounded to cents; 0 when already covered
    total_deposited: Decimal  # PMT × n
    interest_earned: Decimal  # ending balance − current savings − total deposited
    ending_balance: Decimal
    already_covered: bool  # PV·(1+r)^n ≥ FV before any rounding: no deposits needed


def _decimal(value: Number, name: str) -> Decimal:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number, not a bool")
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, (int, float, str)):
        try:
            result = Decimal(str(value).strip())
        except ArithmeticError:
            raise ValueError(f"{name} is not a number: {value!r}") from None
    else:
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if not result.is_finite():
        raise ValueError(f"{name} must be a finite number")
    return result


def _cents(value: Decimal, rounding: str = ROUND_HALF_UP) -> Decimal:
    """Round to cents; call inside the high-precision context so large values fit."""
    return value.quantize(_CENT, rounding=rounding)


def validate_goal_inputs(
    target: Number, months: int, annual_rate: Number, current: Number = 0
) -> list[GoalInputProblem]:
    """Every out-of-range input, in a fixed order; empty when all are valid."""
    problems: list[GoalInputProblem] = []
    if _decimal(target, "target") <= 0:
        problems.append(GoalInputProblem("target", "The target amount must be more than $0."))
    if isinstance(months, bool) or not isinstance(months, int):
        raise TypeError("months must be a whole number of months")
    if months < 1:
        problems.append(
            GoalInputProblem("months", "The time horizon must be at least one month in the future.")
        )
    elif months > MAX_MONTHS:
        problems.append(GoalInputProblem("months", "The time horizon can be at most 100 years."))
    rate = _decimal(annual_rate, "annual_rate")
    if rate < MIN_RATE_PERCENT or rate > MAX_RATE_PERCENT:
        problems.append(
            GoalInputProblem("annual_rate", "The annual rate must be between 0% and 50%.")
        )
    if _decimal(current, "current") < 0:
        problems.append(GoalInputProblem("current", "Current savings can't be negative."))
    return problems


def monthly_savings(
    target: Number, months: int, annual_rate: Number, current: Number = 0
) -> SavingsPlan:
    """The end-of-month deposit needed to grow ``current`` to ``target`` in ``months``.

    ``annual_rate`` is a percentage compounded monthly. Raises ``GoalInputError``
    when an input is out of range (target ≤ 0, months outside 1..1200, rate
    outside 0..50 %, current < 0).
    """
    problems = validate_goal_inputs(target, months, annual_rate, current)
    if problems:
        raise GoalInputError(problems)
    fv = _decimal(target, "target")
    pv = _decimal(current, "current")
    rate = _decimal(annual_rate, "annual_rate")
    n = months
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        r = rate / 100 / 12
        growth = (1 + r) ** n
        pv_grown = pv * growth
        covered = pv_grown >= fv
        if covered:
            pmt = Decimal(0)
        elif r == 0:
            pmt = _cents((fv - pv) / n, ROUND_CEILING)
        else:
            pmt = _cents((fv - pv_grown) * r / (growth - 1), ROUND_CEILING)
        annuity = Decimal(n) if r == 0 else (growth - 1) / r
        ending = _cents(pv_grown + pmt * annuity)
        deposited = _cents(pmt * n)
        interest = _cents(ending - pv - deposited)
        pv_grown_cents = _cents(pv_grown)
    return SavingsPlan(
        target=fv,
        months=n,
        annual_rate=rate,
        current=pv,
        monthly_rate=r,
        growth_factor=growth,
        current_future_value=pv_grown_cents,
        monthly_amount=pmt,
        total_deposited=deposited,
        interest_earned=interest,
        ending_balance=ending,
        already_covered=covered,
    )


def months_until(target_date: date, today: date) -> int:
    """Whole months from ``today`` to ``target_date`` (zero or negative if not in the future).

    A month counts once its day-of-month is reached; a target on the last day of
    its month counts that month in full (so "by June 2030", resolved to June 30,
    is the calendar-month difference).
    """
    months = (target_date.year - today.year) * 12 + (target_date.month - today.month)
    last_day = calendar.monthrange(target_date.year, target_date.month)[1]
    if target_date.day < today.day and target_date.day != last_day:
        months -= 1
    return months


def end_of_month(year: int, month: int) -> date:
    """The last day of ``year``-``month``."""
    return date(year, month, calendar.monthrange(year, month)[1])


__all__ = [
    "GoalInputError",
    "GoalInputProblem",
    "MAX_MONTHS",
    "MAX_RATE_PERCENT",
    "MIN_RATE_PERCENT",
    "SavingsPlan",
    "end_of_month",
    "monthly_savings",
    "months_until",
    "validate_goal_inputs",
]
