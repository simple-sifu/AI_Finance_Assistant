"""Parsing the real knowledge-base articles and malformed front matter."""

from __future__ import annotations

import pytest

from finance_assistant.knowledge import ArticleFormatError, load_articles, parse_article


def test_all_76_articles_parse_with_complete_metadata() -> None:
    articles = load_articles()
    assert len(articles) == 76
    assert len({a.slug for a in articles}) == 76
    for a in articles:
        assert a.title and a.source and a.category and a.license
        assert a.url.startswith("https://")
        assert a.body and not a.body.startswith("---")
    wiki = [a for a in articles if a.source == "Wikipedia"]
    assert len(wiki) == 10
    assert all("CC BY-SA" in a.license and a.source_name == "Wikipedia" for a in wiki)


def test_source_name_is_short_publisher() -> None:
    article = {a.slug: a for a in load_articles()}["what-is-compound-interest"]
    assert article.source_name == "Investor.gov"
    assert article.title == "What is compound interest?"


def test_parse_minimal_article() -> None:
    text = '---\ntitle: "T"\nsource: "S (Long)"\nurl: "https://x"\nlicense: "L"\ncategory: "c"\n---\n\n# T\n\nBody.\n'
    a = parse_article(text, "t")
    assert (a.title, a.source_name, a.url, a.body) == ("T", "S", "https://x", "# T\n\nBody.")


@pytest.mark.parametrize(
    "text",
    [
        "# no front matter",
        "---\ntitle: x\n",
        "---\n- a list\n---\nbody",
        '---\ntitle: "T"\nsource: "S"\nlicense: "L"\ncategory: "c"\n---\nbody',  # no url
        "---\ntitle: [unclosed\n---\nbody",
    ],
)
def test_malformed_front_matter_raises(text: str) -> None:
    with pytest.raises(ArticleFormatError):
        parse_article(text, "bad")


def test_front_matter_fences_must_be_exact_dash_lines() -> None:
    text = (
        '---\ntitle: "T"\nsource: "S"\nurl: "https://x"\nlicense: "L"\ncategory: "c"\n---\n\n'
        "Intro.\n\n----\n\nAfter a rule.\n"
    )
    a = parse_article(text, "t")
    assert a.body == "Intro.\n\n----\n\nAfter a rule."
    with pytest.raises(ArticleFormatError, match="missing front matter"):
        parse_article("----\n" + text[4:], "t")
    # A "----" line inside the header is not a closing fence.
    with pytest.raises(ArticleFormatError, match="unterminated"):
        parse_article('---\ntitle: "T"\n----\nbody\n', "t")
