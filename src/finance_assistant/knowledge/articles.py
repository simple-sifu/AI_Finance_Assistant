"""Load the curated knowledge-base articles (Markdown with YAML front matter)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTICLES_DIR = PROJECT_ROOT / "knowledge_base" / "articles"

_FENCE_RE = re.compile(r"^---[ \t]*\r?\n?$")  # a line that is exactly "---"

REQUIRED_FIELDS = ("title", "source", "url", "license", "category")


class ArticleFormatError(ValueError):
    """An article file has missing or malformed front matter."""


@dataclass(frozen=True)
class Article:
    """One knowledge-base article."""

    slug: str
    title: str
    source: str
    url: str
    license: str
    category: str
    body: str

    @property
    def source_name(self) -> str:
        return source_name(self.source)


def source_name(source: str) -> str:
    """Short publisher name, e.g. "Investor.gov" from "Investor.gov (U.S. SEC)"."""
    return source.split(" (", 1)[0].strip() or source


def parse_article(text: str, slug: str) -> Article:
    """Parse one article's text; raise ArticleFormatError if the front matter is unusable."""
    text = text.lstrip("﻿")
    lines = text.splitlines(keepends=True)
    if not lines or not _FENCE_RE.match(lines[0]):
        raise ArticleFormatError(f"{slug}: missing front matter")
    close = next((i for i in range(1, len(lines)) if _FENCE_RE.match(lines[i])), None)
    if close is None:
        raise ArticleFormatError(f"{slug}: unterminated front matter")
    header, body = "".join(lines[1:close]), "".join(lines[close + 1 :])
    try:
        meta = yaml.safe_load(header)
    except yaml.YAMLError as exc:
        raise ArticleFormatError(f"{slug}: invalid YAML front matter") from exc
    if not isinstance(meta, dict):
        raise ArticleFormatError(f"{slug}: front matter is not a mapping")
    missing = [f for f in REQUIRED_FIELDS if not str(meta.get(f) or "").strip()]
    if missing:
        raise ArticleFormatError(f"{slug}: front matter missing {missing}")
    return Article(
        slug=slug,
        title=str(meta["title"]).strip(),
        source=str(meta["source"]).strip(),
        url=str(meta["url"]).strip(),
        license=str(meta["license"]).strip(),
        category=str(meta["category"]).strip(),
        body=body.strip(),
    )


def article_paths(directory: Path | None = None) -> list[Path]:
    return sorted((directory or DEFAULT_ARTICLES_DIR).glob("*.md"))


def load_articles(directory: Path | None = None) -> list[Article]:
    """Load every ``*.md`` article in ``directory`` (default: knowledge_base/articles), sorted by slug."""
    return [parse_article(p.read_text(encoding="utf-8"), p.stem) for p in article_paths(directory)]


def articles_fingerprint(directory: Path | None = None) -> str:
    """Hash of the article files' names and contents, to detect a stale built index."""
    digest = hashlib.sha256()
    for path in article_paths(directory):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
