"""Pure helpers for the Streamlit app: no Streamlit imports, so they unit-test offline.

- ``escape_markdown_dollars``: stop Streamlit rendering text between ``$`` signs as LaTeX.
- ``source_links``: a reply's sources as Markdown links (http/https only).
- ``error_message``: a readable message for every error the tutor can raise.
- ``to_chat_turns`` / ``Message``: the session's chat history and the tutor's ``ChatTurn``.
- ``goal_question``: the Goals form values, in words, for the Goal Planning agent.
- ``holdings_rows``: an uploaded portfolio as preview table rows.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from urllib.parse import quote, urlsplit

from ..config import ConfigurationError
from ..portfolio import HoldingsFormatError, Portfolio
from ..tutor.models import ChatTurn, Role, Source, TutorReply

logger = logging.getLogger(__name__)

GENERIC_ERROR_TEXT = "Something went wrong while answering. Please try again in a moment."

# A ``$`` after an even number of backslashes (zero included) is not escaped yet.
_UNESCAPED_DOLLAR = re.compile(r"(?<!\\)((?:\\\\)*)\$")
# Fenced blocks and inline code spans: Markdown shows them verbatim, so ``$`` there needs no escape.
_CODE = re.compile(r"(```[\s\S]*?```|`[^`\n]*`)")
_LINK_TEXT_SPECIALS = re.compile(r"([\\\[\]*_`<>])")


def escape_markdown_dollars(text: str) -> str:
    """Escape every ``$`` not already escaped, so amounts display literally (no LaTeX).

    Code spans and fenced blocks are left as is (an escape there would show a backslash).
    """
    parts = _CODE.split(text)
    return "".join(part if i % 2 else _UNESCAPED_DOLLAR.sub(r"\1\\$", part) for i, part in enumerate(parts))


def _link_text(title: str) -> str:
    text = " ".join(title.split()) or "Source"
    return escape_markdown_dollars(_LINK_TEXT_SPECIALS.sub(r"\\\1", text))


def _safe_url(url: str | None) -> str | None:
    if not url:
        return None
    url = url.strip()
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        return None
    # Keep the URL intact but make it safe inside Markdown's (...) link target.
    return quote(url, safe=":/?#[]@!&'*+,;=%~-._")


def source_links(sources: Iterable[Source]) -> list[str]:
    """One Markdown item per source: a clickable link when it has an http(s) URL, else the title."""
    items: list[str] = []
    for source in sources:
        text = _link_text(source.title)
        url = _safe_url(source.url)
        items.append(f"[{text}]({url})" if url else text)
    return items


def error_message(exc: BaseException) -> str:
    """A readable Markdown message for an error raised while asking; never a traceback."""
    if isinstance(exc, HoldingsFormatError):
        problems = [str(p) for p in exc.problems] or [str(exc)]
        lines = "\n".join(f"- {escape_markdown_dollars(p)}" for p in problems)
        return f"Your holdings file has problems, so nothing was sent:\n\n{lines}"
    if isinstance(exc, ConfigurationError):
        return escape_markdown_dollars(str(exc))
    if isinstance(exc, ValueError):
        text = str(exc).strip()
        return escape_markdown_dollars(f"That couldn't be answered: {text}") if text else GENERIC_ERROR_TEXT
    logger.warning("Unexpected error while answering (%s)", type(exc).__name__)
    return GENERIC_ERROR_TEXT


@dataclass(frozen=True)
class Message:
    """One message shown in a tab's conversation (kept in ``st.session_state`` only)."""

    role: Role
    content: str
    sources: tuple[Source, ...] = field(default_factory=tuple)

    @classmethod
    def from_reply(cls, reply: TutorReply) -> Message:
        return cls("assistant", reply.text, tuple(reply.sources))


def to_chat_turns(messages: Sequence[Message]) -> list[ChatTurn]:
    """The conversation so far as the tutor's history."""
    return [ChatTurn(m.role, m.content) for m in messages]


def _amount(value: float | Decimal) -> str:
    number = Decimal(str(value))
    if number == number.to_integral_value():
        return f"${number:,.0f}"
    return f"${number:,.2f}"


def _number(value: float | Decimal) -> str:
    return f"{Decimal(str(value)).normalize():f}"


def goal_question(
    target: float | Decimal,
    years: int,
    annual_rate_percent: float | Decimal | None,
    current_savings: float | Decimal | None,
) -> str:
    """The Goals form's values as a question the Goal Planning agent can read."""
    unit = "year" if years == 1 else "years"
    parts = [f"I want to reach {_amount(target)} in {years} {unit}."]
    if annual_rate_percent is not None:
        parts.append(f"Assume an expected annual return of {_number(annual_rate_percent)}%.")
    if current_savings:
        parts.append(f"I currently have {_amount(current_savings)} saved.")
    else:
        parts.append("I have nothing saved toward it yet.")
    parts.append("How much would I need to save each month?")
    return " ".join(parts)


def holdings_rows(portfolio: Portfolio) -> list[dict[str, str]]:
    """Preview rows for an uploaded portfolio: ticker, shares, value (blank when not given)."""
    rows = []
    for h in portfolio.holdings:
        rows.append(
            {
                "Ticker": h.ticker,
                "Shares": f"{h.shares.normalize():f}" if h.shares is not None else "",
                "Value": _amount(h.value) if h.value is not None else "",
            }
        )
    return rows


__all__ = [
    "GENERIC_ERROR_TEXT",
    "Message",
    "error_message",
    "escape_markdown_dollars",
    "goal_question",
    "holdings_rows",
    "source_links",
    "to_chat_turns",
]
