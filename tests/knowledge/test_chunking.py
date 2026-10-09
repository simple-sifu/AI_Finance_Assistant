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


_TABLE = """\
Intro paragraph.

| Issue | Roth 401(k) | Roth IRA | Pre-tax 401(k) |
| --- | --- | --- | --- |
| Income limits | No income limitation. | Income limits: 2024 single $161,000 | No income limitation. |
| Required distributions | Age 73. | None while alive. | |

After the table.
"""


def test_table_cells_are_labelled_with_row_and_column() -> None:
    (chunk,) = chunk_article(_article(_TABLE))
    assert "Income limits (Roth 401(k)): No income limitation." in chunk.text
    assert "Income limits (Roth IRA): Income limits: 2024 single $161,000" in chunk.text
    assert "Required distributions (Roth IRA): None while alive." in chunk.text
    assert "Required distributions (Pre-tax 401(k))" not in chunk.text  # empty cell dropped
    assert "|" not in chunk.text
    assert chunk.text.startswith("Intro paragraph.") and chunk.text.endswith("After the table.")


def test_table_split_across_chunks_keeps_labels() -> None:
    rows = "\n".join(f"| Row {i} | {'word ' * 30}| {'text ' * 30}|" for i in range(12))
    body = f"| Issue | Account A | Account B |\n| --- | --- | --- |\n{rows}\n"
    chunks = chunk_article(_article(body), chunk_chars=300, overlap_chars=0)
    assert len(chunks) > 1
    for chunk in chunks:
        for line in filter(None, chunk.text.splitlines()):
            assert line.startswith("Row ") and ("(Account A): " in line or "(Account B): " in line)


def test_table_with_empty_header_uses_first_row_as_header() -> None:
    body = "|  |  |  |\n| --- | --- | --- |\n|  | **Fixed** | **Variable** |\n| Growth? | Fixed rate. | Market based. |\n"
    (chunk,) = chunk_article(_article(body))
    assert chunk.text == "Growth? (**Fixed**): Fixed rate.\n\nGrowth? (**Variable**): Market based."


def test_single_column_table_keeps_cell_text() -> None:
    body = "|  |\n| --- |\n| The Rule of 72 estimates doubling time. |\n"
    (chunk,) = chunk_article(_article(body))
    assert chunk.text == "The Rule of 72 estimates doubling time."


def test_pipe_text_without_separator_is_not_a_table() -> None:
    body = "| not a table |\nplain text"
    (chunk,) = chunk_article(_article(body))
    assert chunk.text == "| not a table |\nplain text"


def test_overlap_never_starts_inside_a_table_cell() -> None:
    cells = "\n".join(f"| Row {i} | 2024 - limit ${i}00,000 - 2021 - limit ${i}0,000 for row {i} |" for i in range(8))
    body = f"| Issue | Roth IRA |\n| --- | --- |\n{cells}\n"
    chunks = chunk_article(_article(body), chunk_chars=200, overlap_chars=60)
    assert len(chunks) > 1
    for chunk in chunks:
        for paragraph in chunk.text.split("\n\n"):
            assert paragraph.startswith("Row ") and "(Roth IRA): 2024 - limit" in paragraph


def test_cell_longer_than_a_chunk_repeats_its_label() -> None:
    long_cell = " ".join(f"limit{i} is ${i},000." for i in range(40))
    body = f"| Issue | Roth IRA |\n| --- | --- |\n| Income limits | {long_cell} |\n"
    chunks = chunk_article(_article(body), chunk_chars=300, overlap_chars=0)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.startswith("Income limits (Roth IRA): ")
        assert len(chunk.text) <= 300
