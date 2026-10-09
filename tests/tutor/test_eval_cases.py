"""Offline checks on the shape of the routing and advice evaluation set."""

from __future__ import annotations

import pytest

from finance_assistant.tutor.models import AGENT_ROUTES, CLASSIFIER_ROUTES
from finance_assistant.tutor.router import seeks_advice_keywords

from .eval_cases import CASES

MIN_PLAIN_PER_AGENT = 6
MIN_ADVICE_PER_AGENT = 2
MIN_CLARIFY = 3


def _single_route(route: str, *, seeks_advice: bool):
    return [c for c in CASES if c.routes == (route,) and c.seeks_advice is seeks_advice and not c.history]


@pytest.mark.parametrize("route", AGENT_ROUTES)
def test_each_agent_has_enough_cases(route: str) -> None:
    assert len(_single_route(route, seeks_advice=False)) >= MIN_PLAIN_PER_AGENT
    assert len(_single_route(route, seeks_advice=True)) >= MIN_ADVICE_PER_AGENT


@pytest.mark.parametrize("route", AGENT_ROUTES)
def test_each_agent_has_one_end_to_end_advice_case(route: str) -> None:
    cases = [c for c in CASES if c.end_to_end and c.routes == (route,)]
    assert len(cases) == 1
    assert cases[0].seeks_advice


def test_end_to_end_cases_are_advice() -> None:
    assert all(c.seeks_advice for c in CASES if c.end_to_end)
    assert any(c.end_to_end and len(c.routes) > 1 for c in CASES)


def test_clarify_cases() -> None:
    assert len(_single_route("clarify", seeks_advice=False)) >= MIN_CLARIFY


def test_questions_unique_and_routes_valid() -> None:
    questions = [c.question for c in CASES]
    assert len(questions) == len(set(questions))
    for case in CASES:
        assert case.routes and set(case.routes) <= set(CLASSIFIER_ROUTES), case.question


def test_follow_ups_carry_history() -> None:
    assert any(c.history for c in CASES)


def test_news_case_from_story_8_routes_to_news() -> None:
    (case,) = [c for c in CASES if c.question == "Should I buy Tesla after this week's news?"]
    assert case.routes == ("news",) and case.seeks_advice


@pytest.mark.parametrize("case", [c for c in CASES if not c.seeks_advice], ids=lambda c: c.question)
def test_keyword_backstop_never_flags_plain_questions(case) -> None:  # type: ignore[no-untyped-def]
    # A false positive would force the advice redirect onto an ordinary question.
    assert not seeks_advice_keywords(case.question)
