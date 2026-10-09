"""Tavily news search client with a 15-minute in-process cache (CAP-6).

``NewsClient.search(query)`` POSTs to Tavily's ``/search`` endpoint with
``topic="news"`` and returns the recent articles as ``NewsArticle`` objects
(title, URL, outlet domain, excerpt, publish date when known). Only the article
excerpts are requested: never full pages (``include_raw_content``) and never
Tavily's generated ``answer``.

Every failure (no key, an HTTP error status such as 401/429/432/433/5xx, a
network error or timeout, malformed JSON) raises ``NewsUnavailableError``;
nothing stale or bundled is ever served as current news. Successful searches,
including ones with zero results, are cached for ``NEWS_CACHE_TTL_SECONDS``
under the normalized query; failures are not cached.

Like the market-data client, each search uses a short-lived
``httpx.AsyncClient`` (Streamlit calls ``asyncio.run`` per interaction), and
logs carry exception types and status codes only, never the API key.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import httpx

from .config import Settings, get_settings
from .market_data.cache import TTLCache
from .market_data.client import HTTP_TIMEOUT_SECONDS

if TYPE_CHECKING:  # imported lazily: the tutor package imports this module
    from .tutor.models import Source

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
NEWS_CACHE_TTL_SECONDS = 15 * 60

# Search parameters (story 8 design notes). "basic" depth costs 1 credit.
TOPIC = "news"
TIME_RANGE = "week"
MAX_RESULTS = 5
SEARCH_DEPTH = "basic"
CHUNKS_PER_SOURCE = 3
MAX_QUERY_CHARS = 400
# Social-media posts are not news articles; never cite them.
EXCLUDE_DOMAINS = (
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "youtube.com",
    "x.com",
    "twitter.com",
    "reddit.com",
)
# Upper bound on one article's excerpt, to keep the summary prompt small.
MAX_EXCERPT_CHARS = 1500
# Some results (e.g. social-media videos) carry a whole post as the title.
MAX_TITLE_CHARS = 200

_WHITESPACE_RE = re.compile(r"\s+")


class NewsUnavailableError(RuntimeError):
    """News search could not be done right now (no key, HTTP error, timeout, bad response)."""


@dataclass(frozen=True)
class NewsArticle:
    """One news search result."""

    title: str
    url: str
    domain: str
    content: str
    published: date | None = None

    def as_source(self) -> Source:
        """``"Headline (reuters.com, 2026-10-08)"`` with the URL; the date is left out when unknown."""
        from .tutor.models import Source

        detail = self.domain if self.published is None else f"{self.domain}, {self.published.isoformat()}"
        return Source(title=f"{self.title} ({detail})", url=self.url)


def normalize_query(query: str) -> str:
    """Collapse whitespace and clip to ``MAX_QUERY_CHARS`` (the text sent to Tavily)."""
    return _WHITESPACE_RE.sub(" ", query).strip()[:MAX_QUERY_CHARS].strip()


def _cache_key(query: str) -> str:
    return query.casefold()


def _domain(url: str) -> str | None:
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    host = parts.hostname.lower()
    host = host[4:] if host.startswith("www.") else host
    return host or None


def _parse_date(value: object) -> date | None:
    """Tavily's ``published_date``: RFC 2822 ("Wed, 08 Oct 2026 14:00:00 GMT") or ISO 8601."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(text).date()
    except (TypeError, ValueError, IndexError):
        return None


def _clean(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _parse_result(item: object) -> NewsArticle | None:
    """One result, or None if it lacks a usable title, http(s) URL, or excerpt."""
    if not isinstance(item, dict):
        return None
    title, url, content = item.get("title"), item.get("url"), item.get("content")
    if not isinstance(title, str) or not isinstance(url, str) or not isinstance(content, str):
        return None
    title, url, content = _clean(title), url.strip(), _clean(content)
    domain = _domain(url)
    if not title or not content or domain is None:
        return None
    if len(title) > MAX_TITLE_CHARS:
        title = title[:MAX_TITLE_CHARS].rstrip() + "…"
    if len(content) > MAX_EXCERPT_CHARS:
        content = content[:MAX_EXCERPT_CHARS].rstrip() + "…"
    return NewsArticle(
        title=title,
        url=url,
        domain=domain,
        content=content,
        published=_parse_date(item.get("published_date")),
    )


def parse_results(body: object) -> list[NewsArticle]:
    """Articles from a Tavily response body; raise NewsUnavailableError if its shape is wrong."""
    if not isinstance(body, dict):
        raise NewsUnavailableError("response JSON was not an object")
    results = body.get("results")
    if not isinstance(results, list):
        raise NewsUnavailableError("response had no 'results' list")
    articles: list[NewsArticle] = []
    seen: set[str] = set()
    for item in results:
        article = _parse_result(item)
        if article is not None and article.url not in seen:
            seen.add(article.url)
            articles.append(article)
    return articles


class NewsClient:
    """Tavily news search with a TTL cache. Safe across threads and event loops."""

    def __init__(
        self,
        settings: Settings | None = None,
        cache: TTLCache[tuple[NewsArticle, ...]] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._settings = settings if settings is not None else get_settings()
        self._cache: TTLCache[tuple[NewsArticle, ...]] = (
            cache if cache is not None else TTLCache(NEWS_CACHE_TTL_SECONDS, clock=clock)
        )
        if not self._settings.tavily_api_key:
            logger.warning("TAVILY_API_KEY is not set; news questions will get the 'unavailable' reply.")

    @property
    def cache(self) -> TTLCache[tuple[NewsArticle, ...]]:
        return self._cache

    async def search(self, query: str) -> list[NewsArticle]:
        """Recent news articles for ``query`` (possibly none); raise NewsUnavailableError on any failure."""
        text = normalize_query(query)
        if not text:
            raise ValueError("query must not be empty")
        key = _cache_key(text)
        entry = self._cache.get(key)
        if entry is not None and not entry.is_not_found:
            return list(entry.value)  # type: ignore[arg-type]

        articles = await self._fetch(text)
        self._cache.set(key, tuple(articles))
        return articles

    async def _fetch(self, query: str) -> list[NewsArticle]:
        api_key = self._settings.tavily_api_key
        if not api_key:
            raise NewsUnavailableError("TAVILY_API_KEY is not set")
        payload: dict[str, Any] = {
            "query": query,
            "topic": TOPIC,
            "time_range": TIME_RANGE,
            "max_results": MAX_RESULTS,
            "search_depth": SEARCH_DEPTH,
            "chunks_per_source": CHUNKS_PER_SOURCE,
            "exclude_domains": list(EXCLUDE_DOMAINS),
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as http:
                response = await http.post(
                    TAVILY_SEARCH_URL,
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
        except httpx.HTTPError as exc:
            # The message could carry request details; keep only the type.
            raise NewsUnavailableError(f"network error ({type(exc).__name__})") from None

        if response.status_code != 200:
            raise NewsUnavailableError(f"HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError:
            raise NewsUnavailableError("response was not JSON") from None
        return parse_results(body)


_default_client: NewsClient | None = None
_default_client_lock = threading.Lock()


def get_news_client() -> NewsClient:
    """Return the process-wide news client (created on first use from ``get_settings()``)."""
    global _default_client
    with _default_client_lock:
        if _default_client is None:
            _default_client = NewsClient()
        return _default_client


def reset_news_client() -> None:
    """Drop the process-wide news client and its cache; mainly for tests."""
    global _default_client
    with _default_client_lock:
        _default_client = None
