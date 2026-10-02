"""Split articles into ~800-character chunks along headings and paragraphs.

Each chunk stays inside one section (the text under one heading), keeps its
article's title/url, and starts with a short overlap from the previous chunk of
the same section so a sentence cut at a boundary still has context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .articles import Article

CHUNK_CHARS = 800
OVERLAP_CHARS = 150

_HEADING_RE = re.compile(r"^#{1,6}\s+(.*\S)\s*$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """A piece of one article, with what is needed to cite it."""

    article_slug: str
    title: str
    source: str
    url: str
    category: str
    heading: str
    text: str
    position: int  # 0-based order within the article

    @property
    def embed_text(self) -> str:
        """What gets embedded: the article title and section heading give the chunk context."""
        prefix = self.title if not self.heading or self.heading == self.title else f"{self.title} — {self.heading}"
        return f"{prefix}\n{self.text}"


def _sections(body: str) -> list[tuple[str, list[str]]]:
    """(heading, paragraphs) pairs; text before the first heading has heading ""."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            text = "\n".join(paragraph).strip()
            if text:
                sections[-1][1].append(text)
            paragraph.clear()

    for line in body.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            flush()
            sections.append((match.group(1).strip(), []))
        elif not line.strip():
            flush()
        else:
            paragraph.append(line.rstrip())
    flush()
    return [(h, ps) for h, ps in sections if ps]


def _split_long(paragraph: str, limit: int) -> list[str]:
    """Split a paragraph longer than ``limit`` at sentence ends, then at word ends."""
    if len(paragraph) <= limit:
        return [paragraph]
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_RE.split(paragraph):
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:cut].rstrip())
            sentence = sentence[cut:].lstrip()
        if current and len(current) + 1 + len(sentence) > limit:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        pieces.append(current)
    return [p for p in pieces if p]


def _tail(text: str, size: int) -> str:
    """The last ~``size`` characters of ``text``, starting at a word boundary."""
    if len(text) <= size:
        return text
    start = text.find(" ", len(text) - size)
    return text[start + 1 :] if start != -1 else text[-size:]


def chunk_article(
    article: Article, chunk_chars: int | None = None, overlap_chars: int | None = None
) -> list[Chunk]:
    """Chunk one article. Every chunk is at most ``chunk_chars`` plus the overlap and a separator.

    Defaults are read at call time from ``CHUNK_CHARS``/``OVERLAP_CHARS``.
    """
    chunk_chars = CHUNK_CHARS if chunk_chars is None else chunk_chars
    overlap_chars = OVERLAP_CHARS if overlap_chars is None else overlap_chars
    texts: list[tuple[str, str]] = []
    for heading, paragraphs in _sections(article.body):
        pieces = [p for para in paragraphs for p in _split_long(para, chunk_chars)]
        current = ""
        has_new = False  # current holds text not yet emitted (not just the overlap)
        for piece in pieces:
            if has_new and len(current) + 2 + len(piece) > chunk_chars:
                texts.append((heading, current))
                overlap = _tail(current, overlap_chars) if overlap_chars else ""
                current, has_new = overlap, False
            current = f"{current}\n\n{piece}" if current else piece
            has_new = True
        if has_new:
            texts.append((heading, current))
    if not texts and article.title:
        # A body with no text still gets one chunk so the article is indexed.
        texts.append(("", article.title))
    return [
        Chunk(
            article_slug=article.slug,
            title=article.title,
            source=article.source,
            url=article.url,
            category=article.category,
            heading=heading,
            text=text,
            position=i,
        )
        for i, (heading, text) in enumerate(texts)
    ]


def chunk_articles(articles: list[Article], **kwargs: int) -> list[Chunk]:
    return [chunk for article in articles for chunk in chunk_article(article, **kwargs)]
