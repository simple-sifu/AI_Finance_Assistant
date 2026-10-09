"""Pure UI helpers: dollar escaping, source links, error messages, history, goal wording."""

from __future__ import annotations

from decimal import Decimal

import pytest

from finance_assistant.config import ConfigurationError, Settings
from finance_assistant.portfolio import HoldingsFormatError, parse_holdings_csv
from finance_assistant.tutor import ChatTurn, Source, TutorReply
from finance_assistant.ui.helpers import (
    GENERIC_ERROR_TEXT,
    Message,
    error_message,
    escape_markdown_dollars,
    goal_question,
    holdings_rows,
    source_links,
    to_chat_turns,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Save $1,000 a month for $50,000.", r"Save \$1,000 a month for \$50,000."),
        (r"Already \$5 escaped and $6 not", r"Already \$5 escaped and \$6 not"),
        ("no dollars", "no dollars"),
        ("$$", r"\$\$"),
        (r"\\$5", r"\\\$5"),
        (r"\\\$5 stays", r"\\\$5 stays"),
        ("Cost `$5` and $6\n```\nx = $7\n```\n$8", "Cost `$5` and \\$6\n```\nx = $7\n```\n\\$8"),
    ],
)
def test_escape_markdown_dollars(text: str, expected: str) -> None:
    assert escape_markdown_dollars(text) == expected


def test_source_links() -> None:
    links = source_links(
        [
            Source("Compound interest", "https://www.investor.gov/compound"),
            Source("Wiki (finance)", "https://en.wikipedia.org/wiki/Bond_(finance)"),
            Source("[Tricky] $5 *title*", "http://example.com/a b"),
            Source("No URL"),
            Source("Script", "javascript:alert(1)"),
        ]
    )
    assert links[0] == "[Compound interest](https://www.investor.gov/compound)"
    assert links[1] == "[Wiki (finance)](https://en.wikipedia.org/wiki/Bond_%28finance%29)"
    assert links[2] == r"[\[Tricky\] \$5 \*title\*](http://example.com/a%20b)"
    assert links[3] == "No URL"
    assert links[4] == "Script"
    # A title with an escaped dollar: the backslash is doubled, and the $ is escaped again.
    assert source_links([Source(r"Fees \$5", "https://a.example")]) == [r"[Fees \\\$5](https://a.example)"]


def test_error_message_lists_holdings_problems_by_row() -> None:
    with pytest.raises(HoldingsFormatError) as info:
        parse_holdings_csv(b"ticker,shares\nVOO,abc\n,5\n")
    message = error_message(info.value)
    assert "nothing was sent" in message
    assert "- Row 2: shares 'abc' isn't a number" in message
    assert "- Row 3: the ticker is empty" in message


def test_error_message_missing_key_and_other_errors() -> None:
    with pytest.raises(ConfigurationError) as info:
        Settings().require_openai_api_key()
    assert error_message(info.value).startswith("OPENAI_API_KEY is not set")
    assert "question must be a non-empty string" in error_message(ValueError("question must be a non-empty string"))
    assert error_message(RuntimeError("boom sk-secret")) == GENERIC_ERROR_TEXT
    assert "sk-secret" not in error_message(RuntimeError("sk-secret"))


def test_history_round_trip() -> None:
    reply = TutorReply("Answer", "finance_qa", False, [Source("A", "https://a.example")])
    messages = [Message("user", "Q"), Message.from_reply(reply)]
    assert messages[1].sources == (Source("A", "https://a.example"),)
    assert to_chat_turns(messages) == [ChatTurn("user", "Q"), ChatTurn("assistant", "Answer")]


def test_goal_question_words() -> None:
    assert goal_question(50000.0, 5, 6.5, 2500.0) == (
        "I want to reach $50,000 in 5 years. Assume an expected annual return of 6.5%. "
        "I currently have $2,500 saved. How much would I need to save each month?"
    )
    assert goal_question(Decimal("1234.5"), 1, None, 0) == (
        "I want to reach $1,234.50 in 1 year. I have nothing saved toward it yet. "
        "How much would I need to save each month?"
    )
    assert "return of 7%" in goal_question(1000, 2, 7.0, None)


def test_holdings_rows() -> None:
    portfolio = parse_holdings_csv(b"ticker,shares,value\nVOO,10,\nAAPL,,\"$7,250.00\"\n")
    assert holdings_rows(portfolio) == [
        {"Ticker": "VOO", "Shares": "10", "Value": ""},
        {"Ticker": "AAPL", "Shares": "", "Value": "$7,250"},
    ]
