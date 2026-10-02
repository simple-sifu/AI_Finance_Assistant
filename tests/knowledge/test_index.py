"""Build, save/load round trip, staleness checks, search ranking, and the lazy accessor."""

from __future__ import annotations

import logging
import stat
from pathlib import Path

import pytest

from finance_assistant.knowledge import (
    IndexFormatError,
    KnowledgeIndex,
    get_index,
    index_fingerprint,
    load_or_build_index,
    reset_index,
)
from finance_assistant.knowledge import chunking
from finance_assistant.knowledge import index as index_module

from .fakes import HashEmbedder


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What is compound interest?", {"compound-interest", "what-is-compound-interest"}),
        ("What is the rule of 72?", {"rule-of-72"}),
        ("How do index funds work?", {"index-funds"}),
        ("What is a Roth IRA?", {"roth-iras", "traditional-and-roth-iras", "roth-comparison-chart"}),
    ],
)
def test_search_ranks_the_matching_article_first(fake_index: KnowledgeIndex, query: str, expected: set[str]) -> None:
    hits = fake_index.search(query, 5)
    assert len(hits) == 5
    assert hits[0].chunk.article_slug in expected
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert 0 < hits[0].score <= 1.0001


def test_index_covers_all_articles(fake_index: KnowledgeIndex) -> None:
    assert fake_index.article_count == 76
    assert fake_index.fingerprint == index_fingerprint()


def test_empty_query_returns_nothing(fake_index: KnowledgeIndex) -> None:
    assert fake_index.search("   ", 5) == []


def test_save_load_round_trip(fake_index: KnowledgeIndex, tmp_path: Path) -> None:
    fake_index.save(tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {"index.faiss", "chunks.json"}
    for path in tmp_path.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o644
    embedder = HashEmbedder()
    loaded = KnowledgeIndex.load(tmp_path, embedder, index_fingerprint())
    assert loaded.chunks == fake_index.chunks
    query = "difference between ETFs and mutual funds"
    assert [(h.chunk, round(h.score, 5)) for h in loaded.search(query, 6)] == [
        (h.chunk, round(h.score, 5)) for h in fake_index.search(query, 6)
    ]
    assert embedder.calls == [1]  # loading embeds nothing; only the query


def test_load_rejects_other_model_other_articles_and_corruption(fake_index: KnowledgeIndex, tmp_path: Path) -> None:
    fake_index.save(tmp_path)

    class OtherModel(HashEmbedder):
        name = "other-model"

    with pytest.raises(IndexFormatError, match="embedding model"):
        KnowledgeIndex.load(tmp_path, OtherModel())
    with pytest.raises(IndexFormatError, match="different articles"):
        KnowledgeIndex.load(tmp_path, HashEmbedder(), "stale-fingerprint")
    (tmp_path / "index.faiss").write_bytes(b"garbage")
    with pytest.raises(IndexFormatError):
        KnowledgeIndex.load(tmp_path, HashEmbedder())
    with pytest.raises(IndexFormatError):
        KnowledgeIndex.load(tmp_path / "missing", HashEmbedder())


def test_missing_index_is_built_once_saved_then_reused(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    index_dir = tmp_path / "idx"
    embedder = HashEmbedder()
    with caplog.at_level(logging.INFO, logger="finance_assistant.knowledge.index"):
        first = get_index(embedder=embedder, index_dir=index_dir)
        second = get_index()
    assert first is second
    assert (index_dir / "index.faiss").is_file()
    assert len(embedder.calls) == 1  # all chunks embedded in one build, once
    assert sum("building it from the articles" in r.getMessage() for r in caplog.records) == 1

    reset_index()  # a new process: loads from disk without re-embedding the articles
    embedder2 = HashEmbedder()
    reloaded = get_index(embedder=embedder2, index_dir=index_dir)
    assert reloaded is not first
    assert embedder2.calls == []
    assert len(reloaded.chunks) == len(first.chunks)


def test_unwritable_index_dir_keeps_index_in_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(self, directory):  # type: ignore[no-untyped-def]
        raise PermissionError("read-only")

    monkeypatch.setattr(KnowledgeIndex, "save", fail)
    index = load_or_build_index(HashEmbedder(), tmp_path / "ro")
    assert index.article_count == 76


def test_default_index_dir_is_isolated_in_tests() -> None:
    assert "knowledge_base" not in str(index_module.DEFAULT_INDEX_DIR)


@pytest.mark.parametrize("setting", ["CHUNK_CHARS", "OVERLAP_CHARS"])
def test_changing_chunking_settings_forces_a_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setting: str
) -> None:
    first = load_or_build_index(HashEmbedder(), tmp_path)
    reuse = HashEmbedder()
    load_or_build_index(reuse, tmp_path)
    assert reuse.calls == []  # same settings: loaded from disk

    monkeypatch.setattr(chunking, setting, getattr(chunking, setting) + 100)
    rebuild = HashEmbedder()
    rebuilt = load_or_build_index(rebuild, tmp_path)
    assert len(rebuild.calls) == 1  # all chunks re-embedded
    assert rebuilt.fingerprint != first.fingerprint
