"""Goal Planning agent: every I/O-matrix row offline (respx for OpenAI)."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date

import httpx
import pytest
import respx

from finance_assistant import config
from finance_assistant.config import Settings
from finance_assistant.goals import monthly_savings
from finance_assistant.tutor import (
    ADVICE_REDIRECT,
    AGENT_FAILURE_TEXT,
    DISCLAIMER,
    EDUCATION_SYSTEM_PROMPT,
    AgentRequest,
    ChatTurn,
    Classification,
    GoalPlanningAgent,
    StubAgent,
    ask,
    get_agent,
    install_real_agents,
    reset_agents,
)
from finance_assistant.tutor.goal_planning import (
    GOAL_PLANNING_INSTRUCTIONS,
    INVITE_RATE_TEXT,
    describe_goal,
    fallback_explanation,
    format_illustrative,
    format_plan,
    strip_figure_sentences,
)

from .test_ask import AllowAllReviewer
from .test_router import COMPLETIONS_URL
from .test_router import _completion as completion

SETTINGS = Settings(openai_api_key="sk-test", openai_model="gpt-4o-mini")
TODAY = date(2026, 10, 9)
EXPLANATION = "Each deposit earns interest, and that interest earns interest too, so you deposit less than the target."

FIELDS = (
    "target_amount",
    "horizon_years",
    "horizon_months",
    "target_year",
    "target_month",
    "target_day",
    "annual_rate_percent",
    "current_savings",
)


class FakeOpenAI:
    """respx side effect for both LLM calls: goal extraction (structured) and the explanation."""

    def __init__(self, explanation: str | Callable[[str], str] = EXPLANATION, **values: object) -> None:
        self.values = {name: values.get(name) for name in FIELDS}
        self.explanation = explanation
        self.extractions: list[dict] = []
        self.explains: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "response_format" in body:
            self.extractions.append(body)
            content = json.dumps(self.values)
        else:
            self.explains.append(body)
            prompt = body["messages"][-1]["content"]
            content = self.explanation(prompt) if callable(self.explanation) else self.explanation
        return httpx.Response(200, json=completion(content, body["model"]))

    @property
    def explain_system(self) -> str:
        return self.explains[0]["messages"][0]["content"]

    @property
    def explain_prompt(self) -> str:
        return self.explains[0]["messages"][-1]["content"]


@pytest.fixture
def router():
    with respx.mock(assert_all_called=False) as mock:
        yield mock


def agent() -> GoalPlanningAgent:
    return GoalPlanningAgent(settings=SETTINGS, today=lambda: TODAY)


async def run(router, question: str, seeks_advice: bool = False, history=(), **kwargs) -> tuple[str, FakeOpenAI]:
    llm = FakeOpenAI(**kwargs)
    router.post(COMPLETIONS_URL).mock(side_effect=llm)
    result = await agent().run(AgentRequest(question, tuple(history), seeks_advice))
    assert result.sources == []
    assert DISCLAIMER not in result.text
    return result.text, llm


FULL = {"target_amount": 50000, "horizon_years": 5, "annual_rate_percent": 5}
FULL_Q = "How much do I need to save monthly to reach $50,000 in 5 years at 5%?"


# --- Matrix: full inputs ---------------------------------------------------------------


async def test_full_inputs_show_amount_formula_totals_then_explanation(router) -> None:
    text, llm = await run(router, FULL_Q, **FULL)

    plan = monthly_savings(50000, 60, 5)
    assert text == f"{format_plan(plan)}\n\n{EXPLANATION}"
    assert text.startswith("**Monthly savings needed: $735.23**")
    assert "`PMT = (FV − PV·(1+r)^n) · r / ((1+r)^n − 1)`" in text
    assert "- (1+r)^n = (1 + 0.00416667)^60 = 1.283359" in text
    assert (
        "- PMT = ($50,000.00 − $0.00 × 1.283359) × 0.00416667 / (1.283359 − 1) = **$735.23**" in text
    )
    assert "- Total deposited: $44,113.80 (60 × $735.23)" in text
    assert "- Interest earned: $5,886.31" in text
    assert "- Ending balance: $50,000.11" in text
    assert "- (Ends $0.11 above the target because the monthly amount is rounded up to the cent.)" in text
    assert "- The annual rate is treated as a yearly rate compounded monthly" in text and "not as an APY" in text
    assert ADVICE_REDIRECT not in text
    # Explanation: built with build_system_prompt, and the model sees words, never figures.
    assert llm.explain_system.startswith(EDUCATION_SYSTEM_PROMPT)
    assert GOAL_PLANNING_INSTRUCTIONS.splitlines()[0] in llm.explain_system
    description = llm.explain_prompt.split("Question:")[0]
    assert not any(ch.isdigit() for ch in description)
    assert "Interest earned makes up a moderate part of the ending balance" in description
    assert f"Question: {FULL_Q}" in llm.explain_prompt
    assert llm.extractions[0]["response_format"]["type"] == "json_schema"
    assert llm.extractions[0]["messages"][-1]["content"].endswith(f"Latest question: {FULL_Q}")


async def test_horizon_in_months_and_fractional_years(router) -> None:
    text, _ = await run(router, "$50k in 18 months at 5%", target_amount=50000, horizon_months=18, annual_rate_percent=5)
    assert "- Time (n): 18 months" in text
    text, _ = await run(router, "$50k in 2.5 years at 5%", target_amount=50000, horizon_years=2.5, annual_rate_percent=5)
    assert "- Time (n): 30 months" in text
    text, _ = await run(
        router, "$50k in 2 years 6 months at 5%", target_amount=50000, horizon_years=2, horizon_months=6, annual_rate_percent=5
    )
    assert "- Time (n): 30 months" in text


# --- Matrix: with current savings --------------------------------------------------------


async def test_current_savings_grow_and_reduce_the_amount(router) -> None:
    text, llm = await run(router, FULL_Q + " I already have $10,000 saved", current_savings=10000, **FULL)

    assert text.startswith("**Monthly savings needed: $546.52**")
    assert "- Current savings (PV): $10,000.00" in text
    assert "- PV·(1+r)^n = $10,000.00 × 1.283359 = $12,833.59" in text
    assert "- Current savings: $10,000.00, growing to $12,833.59" in text
    assert "current savings count toward the target and also grow" in llm.explain_prompt


# --- Matrix: target date -------------------------------------------------------------------


async def test_target_date_is_converted_to_months_from_today(router) -> None:
    text, llm = await run(
        router, "$20k by June 2030 at 4%", target_amount=20000, target_year=2030, target_month=6, annual_rate_percent=4
    )
    plan = monthly_savings(20000, 44, 4)
    assert text.startswith(f"**Monthly savings needed: ${plan.monthly_amount:,.2f}**")
    assert "- Time (n): 44 months, from today (2026-10-09) to June 2030" in text
    assert "converted into a whole number of months from today" in llm.explain_prompt


async def test_target_date_with_day_and_year_only(router) -> None:
    text, _ = await run(
        router, "$20k by 2027-03-15 at 4%", target_amount=20000, target_year=2027, target_month=3, target_day=15, annual_rate_percent=4
    )
    assert "- Time (n): 5 months, from today (2026-10-09) to 2027-03-15" in text
    text, _ = await run(router, "$20k by 2030 at 4%", target_amount=20000, target_year=2030, annual_rate_percent=4)
    assert "- Time (n): 50 months, from today (2026-10-09) to the end of 2030" in text


async def test_date_wins_over_a_duration_the_model_derived_itself(router) -> None:
    text, _ = await run(
        router,
        "$20k by June 2030 at 4%",
        target_amount=20000,
        horizon_months=72,  # the model's own (wrong) conversion of the date
        target_year=2030,
        target_month=6,
        annual_rate_percent=4,
    )
    assert "- Time (n): 44 months, from today (2026-10-09) to June 2030" in text


# --- Matrix: zero rate ---------------------------------------------------------------------


async def test_zero_rate_divides_and_earns_no_interest(router) -> None:
    text, llm = await run(router, "$50,000 in 5 years at 0%", target_amount=50000, horizon_years=5, annual_rate_percent=0)
    assert text.startswith("**Monthly savings needed: $833.34**")
    assert "- PMT = ($50,000.00 − $0.00) / 60 = **$833.34**" in text
    assert "- Ending balance: $50,000.40" in text
    assert "- Interest earned: $0.00" in text
    assert "`PMT = (FV − PV) / n`" in text and "((1+r)^n − 1)" not in text
    assert "The rate is zero, so no interest is earned" in llm.explain_prompt


# --- Matrix: already covered ----------------------------------------------------------------


async def test_already_covered_says_no_saving_needed_and_shows_growth(router) -> None:
    text, llm = await run(router, FULL_Q + " I have $40,000", current_savings=40000, **FULL)
    assert text.startswith("**No monthly saving is needed at 5%:** your current savings alone reach the $50,000.00 target")
    assert "PV·(1+r)^n = $40,000.00 × 1.283359 = $51,334.35, which already meets the $50,000.00 target" in text
    assert "rounded up to the cent" not in text
    assert "no monthly deposits are needed" in llm.explain_prompt


async def test_already_covered_at_zero_rate(router) -> None:
    text, llm = await run(
        router, "$10k in 1 year at 0%, I have $10k", target_amount=10000, horizon_years=1, annual_rate_percent=0, current_savings=10000
    )
    assert text.startswith("**No monthly saving is needed at 0%:** your current savings alone reach the $10,000.00 target in 12 months.")
    assert (
        "- With no interest your savings stay at $10,000.00, which already meets the $10,000.00 target, so PMT = **$0.00**"
        in text
    )
    assert "current savings already meet the target" in llm.explain_prompt


# --- Matrix: invalid inputs -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"target_amount": 0, "horizon_years": 5, "annual_rate_percent": 5}, "The target amount must be more than $0."),
        ({"target_amount": -500, "horizon_years": 5, "annual_rate_percent": 5}, "The target amount must be more than $0."),
        ({"target_amount": 5000, "horizon_years": 101, "annual_rate_percent": 5}, "The time horizon can be at most 100 years."),
        ({"target_amount": 5000, "horizon_years": 0, "annual_rate_percent": 5}, "The time horizon must be at least one month"),
        ({"target_amount": 5000, "horizon_years": 5, "annual_rate_percent": -1}, "The annual rate must be between 0% and 50%."),
        ({"target_amount": 5000, "horizon_years": 5, "annual_rate_percent": 51}, "The annual rate must be between 0% and 50%."),
        ({"target_amount": 5000, "horizon_years": 5, "annual_rate_percent": 5, "current_savings": -1}, "Current savings can't be negative."),
        (
            {"target_amount": 5000, "target_year": 2020, "target_month": 6, "annual_rate_percent": 5},
            "The date you gave (June 2020) isn't at least a month from today (2026-10-09).",
        ),
        (
            {"target_amount": 5000, "target_year": 2026, "target_month": 10, "target_day": 20, "annual_rate_percent": 5},
            "The date you gave (2026-10-20) isn't at least a month from today",
        ),
        ({"target_amount": 5000, "target_year": 2200, "annual_rate_percent": 5}, "The time horizon can be at most 100 years."),
        ({"target_amount": 5000, "target_year": 2030, "target_month": 13, "annual_rate_percent": 5}, "I couldn't read the target date."),
    ],
)
async def test_invalid_input_says_which_value_and_skips_explanation(router, values: dict, message: str) -> None:
    text, llm = await run(router, "goal question", **values)
    assert text.startswith("I can't do the math with that value yet:\n- ")
    assert message in text
    assert "Please send the corrected value" in text
    assert "Monthly savings needed" not in text
    assert llm.explains == []


async def test_several_invalid_values_are_all_listed(router) -> None:
    text, llm = await run(router, "q", target_amount=-1, horizon_years=5, annual_rate_percent=80, current_savings=-3)
    lines = text.splitlines()
    assert lines[0] == "I can't do the math with those values yet:"
    assert lines[1:4] == [
        "- The target amount must be more than $0.",
        "- The annual rate must be between 0% and 50%.",
        "- Current savings can't be negative.",
    ]
    assert llm.explains == []


async def test_invalid_value_with_missing_horizon_reports_both(router) -> None:
    text, llm = await run(router, "q", target_amount=-1)
    assert "- The target amount must be more than $0." in text
    assert "- when you want to reach it" in text and "- the annual rate of return" in text
    assert "- your target amount" not in text
    assert llm.explains == []


# --- Matrix: rate missing --------------------------------------------------------------------


async def test_rate_missing_shows_labelled_example_rates_and_invites_own_rate(router) -> None:
    text, llm = await run(router, "How much should I save to have $30k in 4 years?", target_amount=30000, horizon_years=4)

    plans = [monthly_savings(30000, 48, r) for r in (0, 4, 7)]
    assert text == f"{format_illustrative(plans)}\n\n{EXPLANATION}\n\n{INVITE_RATE_TEXT}"
    assert "three **example** annual rates: 0%, 4% and 7%" in text
    assert "not predictions" in text and "not suggestions" in text
    assert "**Example at 0%: $625.00 a month**" in text
    assert "**Example at 4%: $577.38 a month**" in text
    assert "**Example at 7%: $543.39 a month**" in text
    assert "examples only, not predictions or suggestions" in llm.explain_prompt
    assert "at the zero rate it all comes from deposits." in llm.explain_prompt
    assert not any(ch.isdigit() for ch in llm.explain_prompt.split("Question:")[0])


async def test_rate_missing_with_current_savings_can_be_covered_at_an_example_rate(router) -> None:
    text, llm = await run(router, "q", target_amount=10000, horizon_years=10, current_savings=6000)
    assert "**Example at 7%: no monthly saving needed**" in text
    assert "**Example at 0%: $33.34 a month**" in text
    assert "at the zero rate everything beyond the current savings comes from deposits." in llm.explain_prompt
    assert "it all comes from deposits" not in llm.explain_prompt


async def test_rate_missing_covered_even_at_zero_rate(router) -> None:
    _, llm = await run(router, "q", target_amount=10000, horizon_years=10, current_savings=12000)
    assert "at the zero rate the current savings alone already reach the target." in llm.explain_prompt


# --- Matrix: target or horizon missing ----------------------------------------------------------


async def test_everything_missing_asks_for_target_horizon_and_rate(router) -> None:
    text, llm = await run(router, "I want to save for a house")
    assert text == (
        "I can show you the math for a monthly savings goal. To work it out, I need:\n"
        "- your target amount: how much you want to have (for example $50,000)\n"
        "- when you want to reach it: a number of years or months, or a date (such as June 2030)\n"
        "- the annual rate of return to assume, as a percentage (your own assumption; I don't suggest one)\n\n"
        "If you already have some savings toward this goal, tell me how much and I'll include it."
    )
    assert llm.explains == []


async def test_only_horizon_missing_asks_for_exactly_that(router) -> None:
    text, llm = await run(router, "Save $20k at 4%, I have $1,000", target_amount=20000, annual_rate_percent=4, current_savings=1000)
    items = [line for line in text.splitlines() if line.startswith("- ")]
    assert items == ["- when you want to reach it: a number of years or months, or a date (such as June 2030)"]
    assert "If you already have some savings" not in text
    assert llm.explains == []


async def test_only_target_missing_asks_for_target_and_rate(router) -> None:
    text, _ = await run(router, "What do I save over 5 years?", horizon_years=5)
    items = [line.split(":")[0] for line in text.splitlines() if line.startswith("- ")]
    assert items == ["- your target amount", "- the annual rate of return to assume, as a percentage (your own assumption; I don't suggest one)"]


# --- Matrix: advice-flagged ------------------------------------------------------------------------


@pytest.mark.parametrize("model_redirect", [ADVICE_REDIRECT, ADVICE_REDIRECT.replace("'", "’")])
async def test_advice_flagged_but_computable_has_no_redirect(router, model_redirect: str) -> None:
    question = "How much should I save each month to reach $50,000 in 5 years at 5%?"
    text, llm = await run(router, question, seeks_advice=True, explanation=f"{model_redirect} {EXPLANATION}", **FULL)
    assert text.startswith("**Monthly savings needed: $735.23**")
    assert ADVICE_REDIRECT not in text and ADVICE_REDIRECT.replace("'", "’") not in text
    assert text.endswith(EXPLANATION)
    assert "do not open with a refusal" in llm.explain_prompt


async def test_advice_flagged_not_computable_opens_with_redirect_then_asks(router) -> None:
    text, llm = await run(router, "How much should I invest in VOO each month?", seeks_advice=True)
    assert text.startswith(f"{ADVICE_REDIRECT}\n\nI can show you the math for a monthly savings goal.")
    assert "- your target amount" in text and "- when you want to reach it" in text and "- the annual rate" in text
    assert llm.explains == []


async def test_advice_flagged_invalid_input_opens_with_redirect(router) -> None:
    text, _ = await run(router, "q", seeks_advice=True, target_amount=-5, horizon_years=5, annual_rate_percent=5)
    assert text.startswith(f"{ADVICE_REDIRECT}\n\nI can't do the math with that value yet:")


# --- The LLM never retypes a figure ------------------------------------------------------------------


async def test_explanation_sentences_with_digits_are_dropped(router, caplog: pytest.LogCaptureFixture) -> None:
    text, _ = await run(router, FULL_Q, explanation="You need $735.23 a month. Interest does part of the work.", **FULL)
    assert text.endswith("\n\nInterest does part of the work.")
    assert "dropped 1 explanation sentence" in caplog.text


async def test_explanation_sentences_with_spelled_out_figures_are_dropped(router) -> None:
    explanation = (
        "Your goal of fifty thousand dollars is reached in five years. Interest earns interest.\n\n"
        "No one deposits less than the amount shown. The annual rate is divided by twelve, and a zero rate earns nothing."
    )
    text, _ = await run(router, FULL_Q, explanation=explanation, **FULL)
    assert text.endswith(
        "\n\nInterest earns interest.\n\n"
        "No one deposits less than the amount shown. The annual rate is divided by twelve, and a zero rate earns nothing."
    )


def test_strip_figure_sentences() -> None:
    assert strip_figure_sentences("It takes ten years. Often it helps.") == ("Often it helps.", 1)
    assert strip_figure_sentences("Save a few Thousand. Hundreds add up. Fine.") == ("Fine.", 2)
    assert strip_figure_sentences("Costs $5. Words only.") == ("Words only.", 1)


async def test_all_numeric_explanation_falls_back_to_the_description(router) -> None:
    text, _ = await run(router, FULL_Q, explanation="PMT = 735.23.", **FULL)
    plans = [monthly_savings(50000, 60, 5)]
    assert text.endswith(fallback_explanation(plans, illustrative=False, from_date=False))
    assert not any(ch.isdigit() for ch in fallback_explanation(plans, False, False))


def test_descriptions_are_word_only() -> None:
    cases = [
        ([monthly_savings(50000, 60, 5)], False, False),
        ([monthly_savings(50000, 60, 0, 1000)], False, True),
        ([monthly_savings(50000, 60, 5, 40000)], False, False),
        ([monthly_savings(50000, 60, 0, 50000)], False, False),
        ([monthly_savings(30000, 48, r, 500) for r in (0, 4, 7)], True, True),
        ([monthly_savings(1000, 1200, 7)], False, False),  # interest is a large share
    ]
    for plans, illustrative, from_date in cases:
        for line in describe_goal(plans, illustrative, from_date):
            assert not any(ch.isdigit() for ch in line), line
    assert "a large part" in " ".join(describe_goal([monthly_savings(1000, 1200, 7)], False, False))


async def test_follow_up_extraction_sees_history(router) -> None:
    history = (ChatTurn("user", "How much to save for $30k in 4 years?"), ChatTurn("assistant", "Examples at 0%, 4%, 7%"))
    _, llm = await run(router, "Use 6%", history=history, target_amount=30000, horizon_years=4, annual_rate_percent=6)
    prompt = llm.extractions[0]["messages"][-1]["content"]
    assert "User: How much to save for $30k in 4 years?" in prompt
    assert prompt.endswith("Latest question: Use 6%")


# --- Matrix: explanation fails --------------------------------------------------------------------


class _GoalClassifier:
    def __init__(self, seeks_advice: bool = False) -> None:
        self.seeks_advice = seeks_advice

    async def classify(self, question, history):  # type: ignore[no-untyped-def]
        return Classification(route="goal_planning", seeks_advice=self.seeks_advice)


@respx.mock
async def test_explanation_failure_gives_the_graphs_failure_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config.reset_settings()
    install_real_agents()

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "response_format" in body:
            values = {name: FULL.get(name) for name in FIELDS}
            return httpx.Response(200, json=completion(json.dumps(values), body["model"]))
        return httpx.Response(500, json={"error": {"message": "boom"}})

    respx.post(COMPLETIONS_URL).mock(side_effect=respond)

    reply = await ask(FULL_Q, classifier=_GoalClassifier(), reviewer=AllowAllReviewer())

    assert reply.route == "goal_planning"
    assert reply.text == f"{AGENT_FAILURE_TEXT}\n\n{DISCLAIMER}"


@respx.mock
async def test_extraction_failure_gives_the_graphs_failure_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config.reset_settings()
    install_real_agents()
    respx.post(COMPLETIONS_URL).mock(return_value=httpx.Response(500, json={"error": {"message": "boom"}}))

    reply = await ask(FULL_Q, classifier=_GoalClassifier(), reviewer=AllowAllReviewer())

    assert reply.text == f"{AGENT_FAILURE_TEXT}\n\n{DISCLAIMER}"


# --- Registration and ask() ---------------------------------------------------------------------


async def test_install_real_agents_registers_goal_planning_and_reset_restores_stub() -> None:
    install_real_agents()
    assert isinstance(get_agent("goal_planning"), GoalPlanningAgent)
    reset_agents()
    assert type(get_agent("goal_planning")) is StubAgent


@respx.mock
async def test_ask_routes_to_goal_planning_with_review_and_one_disclaimer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config.reset_settings()
    install_real_agents()
    respx.post(COMPLETIONS_URL).mock(side_effect=FakeOpenAI(**FULL))
    reviewed: list[tuple[str, str]] = []

    class RecordingReviewer(AllowAllReviewer):
        async def gives_advice(self, question: str, reply: str) -> bool:
            reviewed.append((question, reply))
            return False

    reply = await ask(FULL_Q, classifier=_GoalClassifier(seeks_advice=True), reviewer=RecordingReviewer())

    assert reply.route == "goal_planning"
    assert reply.text.startswith("**Monthly savings needed: $735.23**")
    assert reply.text.count(DISCLAIMER) == 1 and reply.text.endswith(DISCLAIMER)
    assert len(reviewed) == 1 and reviewed[0][1].startswith("**Monthly savings needed")


@respx.mock
async def test_ask_reviewer_flagging_advice_replaces_the_goal_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config.reset_settings()
    install_real_agents()
    respx.post(COMPLETIONS_URL).mock(side_effect=FakeOpenAI(explanation="You should put it all in VOO.", **FULL))

    class FlagAll(AllowAllReviewer):
        async def gives_advice(self, question: str, reply: str) -> bool:
            return True

    reply = await ask(FULL_Q, classifier=_GoalClassifier(), reviewer=FlagAll())

    assert reply.route == "goal_planning"
    assert reply.text.startswith(ADVICE_REDIRECT)
    assert "$735.23" not in reply.text and "VOO" not in reply.text
