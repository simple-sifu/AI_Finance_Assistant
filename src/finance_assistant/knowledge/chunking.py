"""Split articles into ~800-character chunks along headings and paragraphs.

Each chunk stays inside one section (the text under one heading), keeps its
article's title/url, and starts with a short overlap from the previous chunk of
the same section so a sentence cut at a boundary still has context.

Markdown tables are rewritten first: each cell becomes its own paragraph,
labelled with its row label and column header ("Income limits (Roth IRA): ...").
A chunk cut inside a table then still says which row and column each value
belongs to; the raw rows would lose the header row after the first chunk, and a
reader could attach a cell to the wrong column.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .articles import Article

CHUNK_CHARS = 800
OVERLAP_CHARS = 150
# Bump when chunking output changes for the same articles, so saved indexes are rebuilt.
CHUNKER_VERSION = 2  # 2: tables linearized into labelled cells

_HEADING_RE = re.compile(r"^#{1,6}\s+(.*\S)\s*$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$")
_WHITESPACE_RE = re.compile(r"\s+")
# The "row label (column): " prefix of a linearized table cell.
_CELL_LABEL_RE = re.compile(r"^[^\n:]{1,120} \([^\n:]{1,80}\): ")


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


def _cells(row: str) -> list[str]:
    return [_WHITESPACE_RE.sub(" ", c).strip() for c in row.strip().strip("|").split("|")]


def _table_paragraphs(rows: list[str]) -> list[str]:
    """One paragraph per non-empty data cell, as "row label (column): value"."""
    header, data = _cells(rows[0]), [_cells(r) for r in rows[2:]]
    if not any(header) and data and not data[0][0] and any(data[0][1:]):
        # Some articles leave the header row empty and put the column names in the first
        # row, under an empty corner cell.
        header, data = data[0], data[1:]
    paragraphs = []
    for cells in data:
        label = cells[0] if cells else ""
        lines = [label] if label and len(cells) == 1 else []
        for i, cell in enumerate(cells[1:], start=1):
            if not cell:
                continue
            column = header[i] if i < len(header) else ""
            name = f"{label} ({column})" if label and column else label or column
            lines.append(f"{name}: {cell}" if name else cell)
        paragraphs.extend(lines)
    return paragraphs


def _linearize_tables(body: str) -> str:
    """Replace each Markdown table (header row, separator row, data rows) with labelled paragraphs."""
    lines = body.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        if _TABLE_ROW_RE.match(lines[i]) and i + 1 < len(lines) and _TABLE_SEPARATOR_RE.match(lines[i + 1]):
            end = i + 2
            while end < len(lines) and _TABLE_ROW_RE.match(lines[end]):
                end += 1
            for paragraph in _table_paragraphs(lines[i:end]):
                out.extend(["", paragraph, ""])
            i = end
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


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

    for line in _linearize_tables(body).splitlines():
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


def _split_paragraph(paragraph: str, limit: int) -> list[str]:
    """``_split_long``, except that each later piece of a table cell longer than a chunk repeats its label."""
    label = _CELL_LABEL_RE.match(paragraph)
    if label is None or len(paragraph) <= limit:
        return _split_long(paragraph, limit)
    prefix = f"{label.group(0)}… "
    first, *rest = _split_long(paragraph, max(limit - len(prefix), 1))
    return [first] + [prefix + piece for piece in rest]


def _tail(text: str, size: int) -> str:
    """The last ~``size`` characters of ``text``, starting at a word boundary."""
    if len(text) <= size:
        return text
    start = text.find(" ", len(text) - size)
    return text[start + 1 :] if start != -1 else text[-size:]


def _overlap(current: str, size: int) -> str:
    """The overlap carried into the next chunk, never starting partway through a table cell.

    A labelled cell cut partway loses the context that makes it true (e.g. the year
    in "2021 - modified AGI married $208,000"), so the overlap skips to the next
    whole paragraph instead, or is empty.
    """
    tail = _tail(current, size)
    start = len(current) - len(tail)
    paragraph_start = current.rfind("\n\n", 0, start) + 2 if "\n\n" in current[:start] else 0
    if start <= paragraph_start or _CELL_LABEL_RE.match(current[paragraph_start:]) is None:
        return tail
    next_paragraph = current.find("\n\n", start)
    return current[next_paragraph + 2 :] if next_paragraph != -1 else ""


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
        pieces = [p for para in paragraphs for p in _split_paragraph(para, chunk_chars)]
        current = ""
        has_new = False  # current holds text not yet emitted (not just the overlap)
        for piece in pieces:
            if has_new and len(current) + 2 + len(piece) > chunk_chars:
                texts.append((heading, current))
                overlap = _overlap(current, overlap_chars) if overlap_chars else ""
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
