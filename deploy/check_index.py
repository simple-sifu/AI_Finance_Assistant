"""Build-time check: the saved index loads as-is and answers a search with the cached model, offline.

Calls ``KnowledgeIndex.load`` directly (not ``get_index``, which would quietly
rebuild in memory if the saved index were stale or broken). Run by the
Dockerfile as the unprivileged app user with ``HF_HUB_OFFLINE=1``.
"""

from finance_assistant.knowledge import (
    DEFAULT_INDEX_DIR,
    KnowledgeIndex,
    SentenceTransformerEmbedder,
    index_fingerprint,
)

index = KnowledgeIndex.load(DEFAULT_INDEX_DIR, SentenceTransformerEmbedder(), index_fingerprint())
hits = index.search("What is compound interest?", 3)
assert hits, "offline index search returned nothing"
print(f"Offline index check ok: {index.article_count} articles; top hit {hits[0].chunk.title!r}")
