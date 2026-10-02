from __future__ import annotations

import pytest

from finance_assistant.knowledge import KnowledgeIndex, build_index

from .fakes import HashEmbedder


@pytest.fixture(scope="session")
def fake_index() -> KnowledgeIndex:
    """All 76 real articles indexed with the deterministic fake embedder (built once)."""
    return build_index(HashEmbedder())
