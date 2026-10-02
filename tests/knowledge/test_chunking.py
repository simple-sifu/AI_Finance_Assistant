"""Chunk boundaries, sizes, overlap, and metadata."""

from __future__ import annotations

from finance_assistant.knowledge import CHUNK_CHARS, OVERLAP_CHARS, Article, chunk_article, chunk_articles, load_articles


def _article(body: str) -> Article:
    return Article("slug", "Title", "Src (Long)", "https://example.com/a", "L", "concepts", body)


def test_every_article_chunks_with_its_metadata() -> None:
    articles = load_articles()
    chunks = chunk_articles(articles)
    by_slug = {a.slug: a for a in articles}
    assert {c.article_slug for c in chunks} == set(by_slug)
    for c in chunks:
        a = by_slug[c.article_slug]
        assert (c.title, c.url, c.source, c.category) == (a.title, a.url, a.source, a.category)
        assert c.text.strip()
        assert len(c.text) <= CHUNK_CHARS + OVERLAP_CHARS + 2
        assert not c.text.lstrip().startswith("#")  # headings are metadata, not chunk text


def test_positions_are_sequential_per_article() -> None:
    for article in load_articles()[:10]:
        assert [c.position for c in chunk_article(article)] == list(range(len(chunk_article(article))))


def test_sections_never_share_a_chunk_and_keep_their_heading() -> None:
    body = "Intro paragraph.\n\n## First\n\nAlpha text.\n\n## Second\n\nBeta text."
    chunks = chunk_article(_article(body))
    assert [(c.heading, c.text) for c in chunks] == [
        ("", "Intro paragraph."),
        ("First", "Alpha text."),
        ("Second", "Beta text."),
    ]
    assert chunks[1].embed_text == "Title — First\nAlpha text."


def test_long_section_splits_with_overlap_and_loses_no_text() -> None:
    paragraphs = [f"Paragraph {i} " + " ".join(f"word{i}x{j}." for j in range(40)) for i in range(6)]
    chunks = chunk_article(_article("## Sec\n\n" + "\n\n".join(paragraphs)))
    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:]):
        tail = prev.text[-40:]
        assert tail in nxt.text  # the next chunk starts with the previous chunk's tail
    joined = " ".join(c.text for c in chunks)
    for p in paragraphs:
        for sentence in p.split(". "):
            assert sentence.rstrip(".") in joined


def test_oversized_paragraph_is_split_at_sentences() -> None:
    para = " ".join(f"Sentence number {i} is here." for i in range(80))
    chunks = chunk_article(_article(para))
    assert len(chunks) >= 3
    assert all(len(c.text) <= CHUNK_CHARS + OVERLAP_CHARS + 2 for c in chunks)


def test_empty_body_still_yields_one_chunk() -> None:
    chunks = chunk_article(_article(""))
    assert [c.text for c in chunks] == ["Title"]
