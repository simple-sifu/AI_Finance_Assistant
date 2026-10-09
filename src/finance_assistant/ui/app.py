"""The Streamlit app (CAP-10): five tabs that reach all six agents through ``ask``.

- **Chat**: any question, routed by the router, with history.
- **Portfolio**: CSV upload, holdings preview, questions to the Portfolio agent.
- **Markets**: questions to the Market agent, and news questions to the News agent.
- **Goals**: a savings-goal form whose values go to the Goal Planning agent in words.
- **Knowledge**: the knowledge-base article list, and questions routed by the router.

Every answer comes from ``ask`` (so routing, review and the disclaimer stay as
they are). Single-agent panels pass a ``ScopedClassifier``, which keeps the
router's advice flag but fixes the route. Uploads and conversations live only in
``st.session_state``; nothing is written to disk.

When ``APP_PASSWORD`` is set (the deployed app), a password screen comes first
and nothing else renders until the visitor enters it.

Run with ``uv run streamlit run app.py`` from the repository root.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
from collections.abc import Callable

import streamlit as st

from ..config import get_settings
from ..goals import MAX_MONTHS, MAX_RATE_PERCENT
from ..knowledge.articles import Article, load_articles
from ..portfolio import HoldingsFormatError, Portfolio, parse_holdings_csv
from ..tutor import install_real_agents
from ..tutor.graph import ask
from ..tutor.models import Source
from ..tutor.router import AdviceReviewer, Classifier, OpenAIClassifier, ScopedClassifier
from .helpers import (
    Message,
    error_message,
    escape_markdown_dollars,
    goal_question,
    holdings_rows,
    source_links,
    to_chat_turns,
)

logger = logging.getLogger(__name__)

TAB_NAMES = ("Chat", "Portfolio", "Markets", "Goals", "Knowledge")
PORTFOLIO_FORMAT_HINT = (
    "Upload a CSV with a header row: a `ticker` column and a `shares` or `value` column "
    "(each row gives one of them; a value is the dollar amount you hold). Example:\n\n"
    "```\nticker,shares,value\nVOO,10,\nAAPL,,7250\n```"
)

WRONG_PASSWORD_TEXT = "Incorrect password."

BAD_UPLOAD_TEXT = "Fix the problems in your holdings file before asking; nothing was sent."

# Conversation keys in st.session_state["conversations"].
CHAT, PORTFOLIO, MARKET, NEWS, GOALS, KNOWLEDGE = "chat", "portfolio", "market", "news", "goals", "knowledge"


# -- seams (tests replace these with offline fakes) ----------------------------------


def router_classifier() -> Classifier:
    """The router's classifier. Raises ``ConfigurationError`` when OPENAI_API_KEY is unset."""
    return OpenAIClassifier(get_settings())


def advice_reviewer() -> AdviceReviewer | None:
    """The guardrail's reviewer; ``None`` lets ``ask`` use its default (OpenAI) reviewer."""
    return None


@st.cache_resource(show_spinner=False)
def _install_agents() -> bool:
    install_real_agents()
    return True


@st.cache_data(show_spinner=False)
def _articles() -> list[Article]:
    return load_articles()


# -- asking ---------------------------------------------------------------------------


def _conversation(key: str) -> list[Message]:
    conversations = st.session_state.setdefault("conversations", {})
    return conversations.setdefault(key, [])


def _errors() -> dict[str, str]:
    return st.session_state.setdefault("errors", {})


def _ask(
    key: str,
    question: str,
    *,
    route: str | None = None,
    portfolio: Portfolio | None = None,
    use_history: bool = True,
) -> None:
    """Ask ``question`` and append the exchange to conversation ``key``; errors go to the tab."""
    question = (question or "").strip()
    if not question:
        return
    messages = _conversation(key)
    history = to_chat_turns(messages) if use_history else []
    try:
        classifier = router_classifier()
        if route is not None:
            classifier = ScopedClassifier(route, classifier)
        with st.spinner("Thinking…"):
            reply = asyncio.run(
                ask(question, history, portfolio=portfolio, classifier=classifier, reviewer=advice_reviewer())
            )
    except Exception as exc:  # noqa: BLE001 - every error is shown as a message, never a traceback
        _errors()[key] = f"{error_message(exc)}\n\nYour question: {escape_markdown_dollars(question)}"
        return
    _errors().pop(key, None)
    messages.append(Message("user", question))
    messages.append(Message.from_reply(reply))


def _render_message(message: Message) -> None:
    with st.chat_message(message.role):
        st.markdown(escape_markdown_dollars(message.content))
        links = source_links(message.sources)
        if links:
            st.markdown("**Sources**\n\n" + "\n".join(f"- {link}" for link in links))


def _render_conversation(key: str) -> None:
    for message in _conversation(key):
        _render_message(message)
    error = _errors().get(key)
    if error:
        st.error(error)


def _question_form(key: str, label: str, placeholder: str, submit: str = "Ask") -> str | None:
    """A question box that clears on submit; returns the submitted text (``None`` if not submitted)."""
    with st.form(f"{key}_form", clear_on_submit=True):
        text = st.text_input(label, placeholder=placeholder, key=f"{key}_question")
        submitted = st.form_submit_button(submit)
    return text if submitted else None


# -- tabs -----------------------------------------------------------------------------


def chat_tab() -> None:
    st.caption("Ask anything about investing. Your question goes to the right specialist.")
    history = st.container()
    prompt = st.chat_input("Ask a finance question", key="chat_input")
    if prompt:
        _ask(CHAT, prompt)
    with history:
        _render_conversation(CHAT)


def _uploaded_portfolio() -> tuple[Portfolio | None, bool]:
    """The uploaded portfolio (or None) and whether the upload is unusable."""
    upload = st.file_uploader("Holdings CSV", type=["csv"], key="portfolio_upload")
    if upload is None:
        return None, False
    try:
        portfolio = parse_holdings_csv(upload.getvalue())
    except HoldingsFormatError as exc:
        st.error(error_message(exc))
        return None, True
    st.caption(f"Holdings preview ({len(portfolio.holdings)} holdings)")
    st.dataframe(holdings_rows(portfolio), hide_index=True)
    return portfolio, False


def portfolio_tab() -> None:
    st.caption("Upload your holdings and learn how diversified or concentrated they are.")
    st.markdown(PORTFOLIO_FORMAT_HINT)
    portfolio, bad_upload = _uploaded_portfolio()
    if not bad_upload and _errors().get(PORTFOLIO) == BAD_UPLOAD_TEXT:
        _errors().pop(PORTFOLIO)  # the file was fixed or removed
    history = st.container()
    question = _question_form(PORTFOLIO, "Ask about your portfolio", "How diversified is my portfolio?")
    if question is not None and question.strip():
        if bad_upload:
            _errors()[PORTFOLIO] = BAD_UPLOAD_TEXT
        else:
            _ask(PORTFOLIO, question, route="portfolio", portfolio=portfolio)
    with history:
        _render_conversation(PORTFOLIO)


def markets_tab() -> None:
    st.subheader("Quotes")
    st.caption("Ask about a ticker and what its numbers mean.")
    quotes = st.container()
    question = _question_form(MARKET, "Ticker or question", "What is AAPL trading at?")
    if question is not None:
        _ask(MARKET, question, route="market")
    with quotes:
        _render_conversation(MARKET)

    st.subheader("News")
    st.caption("Ask for a summary of current financial news, with sources.")
    news = st.container()
    question = _question_form(NEWS, "News question", "What happened in the markets today?")
    if question is not None:
        _ask(NEWS, question, route="news")
    with news:
        _render_conversation(NEWS)


def goals_tab() -> None:
    st.caption("See how much you would need to save each month to reach a goal.")
    with st.form("goals_form"):
        target = st.number_input("Target amount ($)", min_value=0.0, value=None, step=1000.0, key="goal_target")
        years = st.number_input(
            "Years to reach it", min_value=1, max_value=MAX_MONTHS // 12, value=None, step=1, key="goal_years"
        )
        rate = st.number_input(
            "Expected annual rate (%) — optional",
            min_value=0.0,
            max_value=float(MAX_RATE_PERCENT),
            value=None,
            step=0.5,
            key="goal_rate",
        )
        current = st.number_input(
            "Current savings ($)", min_value=0.0, value=0.0, step=500.0, key="goal_current"
        )
        submitted = st.form_submit_button("Calculate")
    history = st.container()
    if submitted:
        if not target or not years:
            _errors()[GOALS] = "Enter a target amount and the number of years; nothing was sent."
        else:
            _ask(GOALS, goal_question(target, int(years), rate, current), route="goal_planning", use_history=False)
    with history:
        _render_conversation(GOALS)


def _article_list() -> None:
    try:
        articles = _articles()
    except Exception as exc:  # noqa: BLE001 - a broken article file must not break the tab
        logger.warning("Could not load the knowledge base (%s)", type(exc).__name__)
        st.warning("The knowledge-base article list couldn't be loaded.")
        return
    with st.expander(f"Browse the knowledge base ({len(articles)} articles)"):
        by_category: dict[str, list[Article]] = {}
        for article in articles:
            by_category.setdefault(article.category, []).append(article)
        for category in sorted(by_category):
            st.markdown(f"**{escape_markdown_dollars(category.replace('_', ' ').title())}**")
            lines = []
            for article in sorted(by_category[category], key=lambda a: a.title.lower()):
                link = source_links([Source(article.title, article.url)])[0]
                lines.append(f"- {link} — {escape_markdown_dollars(article.source_name)}")
            st.markdown("\n".join(lines))


def knowledge_tab() -> None:
    st.caption("Learn investing concepts and tax-advantaged accounts from curated articles.")
    _article_list()
    history = st.container()
    question = _question_form(
        KNOWLEDGE, "Ask a concept question", "How is a Roth IRA different from a traditional IRA?"
    )
    if question is not None:
        _ask(KNOWLEDGE, question)
    with history:
        _render_conversation(KNOWLEDGE)


TABS: tuple[Callable[[], None], ...] = (chat_tab, portfolio_tab, markets_tab, goals_tab, knowledge_tab)


def _signed_in() -> bool:
    """True when no password is configured or this session entered it; else show the password screen."""
    password = get_settings().app_password
    if password is None or st.session_state.get("signed_in"):
        return True
    with st.form("password_form", clear_on_submit=True):
        entered = st.text_input("Password", type="password", key="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        # APP_PASSWORD is stripped when loaded, so strip the entry too (pasted spaces).
        if hmac.compare_digest(entered.strip().encode(), password.encode()):
            st.session_state["signed_in"] = True
            st.rerun()
        st.error(WRONG_PASSWORD_TEXT)
    return False


def main() -> None:
    st.set_page_config(page_title="AI Finance Tutor", layout="centered")
    st.title("AI Finance Tutor")
    st.caption("A patient tutor that explains investing. It teaches and never gives personal advice.")
    if not _signed_in():
        st.stop()
    _install_agents()
    for tab, render in zip(st.tabs(list(TAB_NAMES)), TABS, strict=True):
        with tab:
            render()


__all__ = ["TAB_NAMES", "advice_reviewer", "main", "router_classifier"]
