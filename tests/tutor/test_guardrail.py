"""The shared guardrail: disclaimer footer and the shared system-prompt text."""

from __future__ import annotations

from finance_assistant.tutor import (
    DISCLAIMER,
    EDUCATION_SYSTEM_PROMPT,
    apply_guardrail,
    build_system_prompt,
)


def test_disclaimer_wording() -> None:
    assert DISCLAIMER == "*Educational information only, not financial, tax, or investment advice.*"


def test_apply_guardrail_appends_footer_once() -> None:
    once = apply_guardrail("Compound interest is interest on interest.  \n")
    assert once == f"Compound interest is interest on interest.\n\n{DISCLAIMER}"
    assert apply_guardrail(once) == once


def test_apply_guardrail_on_empty_text() -> None:
    assert apply_guardrail("") == DISCLAIMER


def test_build_system_prompt_starts_with_shared_rule() -> None:
    prompt = build_system_prompt("You explain ETFs.")
    assert prompt.startswith(EDUCATION_SYSTEM_PROMPT)
    assert prompt.endswith("You explain ETFs.")
    assert build_system_prompt("  ") == EDUCATION_SYSTEM_PROMPT
    assert "never" in EDUCATION_SYSTEM_PROMPT.lower()


def test_apply_guardrail_removes_mid_text_disclaimer() -> None:
    text = f"Part one.\n{DISCLAIMER}\nPart two."
    assert apply_guardrail(text) == f"Part one.\n\nPart two.\n\n{DISCLAIMER}"


def test_apply_guardrail_collapses_doubled_disclaimer() -> None:
    result = apply_guardrail(f"Answer.\n\n{DISCLAIMER}\n\n{DISCLAIMER}")
    assert result == f"Answer.\n\n{DISCLAIMER}"
    assert result.count(DISCLAIMER) == 1
