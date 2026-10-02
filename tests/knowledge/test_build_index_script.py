"""Smoke test for scripts/build_index.py with the fake embedder."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from finance_assistant.knowledge import KnowledgeIndex
from finance_assistant.knowledge import index as index_module

from .fakes import HashEmbedder

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_index.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_index_script", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_writes_a_loadable_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(index_module, "SentenceTransformerEmbedder", HashEmbedder)
    out = tmp_path / "index"

    assert _load_script().main(["--out", str(out)]) == 0

    assert (out / "index.faiss").is_file() and (out / "chunks.json").is_file()
    loaded = KnowledgeIndex.load(out, HashEmbedder(), index_module.index_fingerprint())
    assert loaded.article_count == 76
    assert capsys.readouterr().out.startswith(f"Indexed 76 articles as {len(loaded.chunks)} chunks into ")
