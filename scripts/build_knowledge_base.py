"""Download the curated knowledge-base articles and save them as Markdown.

Each entry in kb_sources.json becomes one Markdown file with YAML front matter
(title, source, url, license, category, retrieved), so the RAG pipeline can
cite the article it drew an answer from.

Usage:
    pip install -r scripts/requirements-kb.txt
    python scripts/build_knowledge_base.py            # skip articles already saved
    python scripts/build_knowledge_base.py --force    # re-download everything
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCES = ROOT / "scripts" / "kb_sources.json"
DEFAULT_OUT = ROOT / "knowledge_base" / "articles"

# Wikipedia's API policy requires a descriptive User-Agent.
USER_AGENT = "AIFinanceAssistant-KB-Builder/1.0 (educational course project)"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
MIN_WORDS = 50

SITES = {
    "www.investor.gov": {
        "source": "Investor.gov (U.S. Securities and Exchange Commission)",
        "license": "Public domain (U.S. Government work)",
        "title_suffix": " | Investor.gov",
    },
    "www.irs.gov": {
        "source": "IRS.gov (Internal Revenue Service)",
        "license": "Public domain (U.S. Government work)",
        "title_suffix": " | Internal Revenue Service",
    },
}
WIKIPEDIA_META = {
    "source": "Wikipedia",
    "license": "CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)",
}


@dataclass
class Article:
    title: str
    url: str
    source: str
    license: str
    category: str
    body: str

    @property
    def word_count(self) -> int:
        return len(self.body.split())


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:80] or "article"


def clean_markdown(md: str) -> str:
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def get(session: requests.Session, url: str, **kwargs) -> requests.Response:
    for attempt in range(4):
        try:
            resp = session.get(url, timeout=30, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == 3:
                raise
            retry_after = exc.response.headers.get("Retry-After") if exc.response is not None else None
            time.sleep(int(retry_after) if retry_after and retry_after.isdigit() else 5 * (attempt + 1))
    raise AssertionError("unreachable")


def fetch_web_page(session: requests.Session, entry: dict) -> Article:
    url = entry["url"]
    site = SITES.get(urlparse(url).netloc)
    if site is None:
        raise ValueError(f"no extraction rules for {urlparse(url).netloc}")

    soup = BeautifulSoup(get(session, url).text, "html.parser")
    body = soup.select_one("article .field--name-body") or soup.select_one("main article")
    if body is None:
        raise ValueError("could not find the article body")

    for tag in body.select("script, style, noscript, iframe, form, img, figure, button, nav"):
        tag.decompose()

    title = entry.get("title")
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(" ", strip=True) if h1 else soup.title.get_text(strip=True)
        title = title.removesuffix(site["title_suffix"]).strip()

    md = markdownify(str(body), heading_style="ATX", bullets="-", strip=["a"])
    return Article(
        title=title,
        url=url,
        source=site["source"],
        license=site["license"],
        category=entry["category"],
        body=clean_markdown(md),
    )


def fetch_wikipedia(session: requests.Session, entry: dict) -> Article:
    """Fetch only the lead section: it is the most beginner-friendly part."""
    resp = get(
        session,
        WIKIPEDIA_API,
        params={
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "exintro": 1,
            "redirects": 1,
            "titles": entry["title"],
            "format": "json",
            "formatversion": 2,
        },
    )
    page = resp.json()["query"]["pages"][0]
    if page.get("missing"):
        raise ValueError(f"Wikipedia page not found: {entry['title']}")

    title = page["title"]
    return Article(
        title=title,
        url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        source=WIKIPEDIA_META["source"],
        license=WIKIPEDIA_META["license"],
        category=entry["category"],
        body=clean_markdown(page.get("extract", "")),
    )


def to_markdown(article: Article, retrieved: str) -> str:
    def quote(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)

    front_matter = "\n".join(
        [
            "---",
            f"title: {quote(article.title)}",
            f"source: {quote(article.source)}",
            f"url: {quote(article.url)}",
            f"license: {quote(article.license)}",
            f"category: {quote(article.category)}",
            f"retrieved: {retrieved}",
            "---",
        ]
    )
    return f"{front_matter}\n\n# {article.title}\n\n{article.body}\n"


def entry_url(entry: dict) -> str:
    if entry.get("source") == "wikipedia":
        return f"https://en.wikipedia.org/wiki/{entry['title'].replace(' ', '_')}"
    return entry["url"]


def saved_urls(out_dir: Path) -> set[str]:
    urls = set()
    for path in out_dir.glob("*.md"):
        match = re.search(r"^url: (.+)$", path.read_text(), re.MULTILINE)
        if match:
            urls.add(json.loads(match.group(1)))
    return urls


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    parser.add_argument("--force", action="store_true", help="re-download articles that already exist")
    args = parser.parse_args()

    entries = json.loads(args.sources.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    existing = set() if args.force else saved_urls(args.out)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    retrieved = date.today().isoformat()
    saved, skipped, failed = 0, 0, []
    seen_slugs: set[str] = set()
    # Some URLs redirect to the same page; keep only the first copy.
    seen_bodies: dict[str, str] = {}

    for i, entry in enumerate(entries, 1):
        label = entry_url(entry)
        if label in existing:
            skipped += 1
            continue

        try:
            fetch = fetch_wikipedia if entry.get("source") == "wikipedia" else fetch_web_page
            article = fetch(session, entry)
            if article.word_count < MIN_WORDS:
                raise ValueError(f"only {article.word_count} words (min {MIN_WORDS})")
            if article.body in seen_bodies:
                raise ValueError(f"duplicate content of {seen_bodies[article.body]}")
            seen_bodies[article.body] = label
        except Exception as exc:  # keep going; report every failure at the end
            failed.append((label, str(exc)))
            print(f"[{i}/{len(entries)}] FAIL {label}: {exc}", file=sys.stderr)
            continue
        finally:
            time.sleep(args.delay)

        slug = slugify(article.title)
        if slug in seen_slugs:
            slug = f"{slug}-{i}"
        seen_slugs.add(slug)

        (args.out / f"{slug}.md").write_text(to_markdown(article, retrieved))
        saved += 1
        print(f"[{i}/{len(entries)}] saved {slug}.md ({article.word_count} words)")

    total = len(list(args.out.glob("*.md")))
    print(f"\nSaved {saved}, skipped {skipped} existing, failed {len(failed)}. {total} articles in {args.out}")
    for label, reason in failed:
        print(f"  - {label}: {reason}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
