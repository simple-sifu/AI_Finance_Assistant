"""The curated knowledge base: articles, chunks, and the FAISS index used for retrieval."""

from __future__ import annotations

from .articles import (
    DEFAULT_ARTICLES_DIR,
    Article,
    ArticleFormatError,
    articles_fingerprint,
    load_articles,
    parse_article,
    source_name,
)
from .chunking import CHUNK_CHARS, OVERLAP_CHARS, Chunk, chunk_article, chunk_articles
from .index import (
    DEFAULT_INDEX_DIR,
    EMBEDDING_MODEL,
    Embedder,
    Hit,
    IndexFormatError,
    KnowledgeIndex,
    SentenceTransformerEmbedder,
    build_index,
    get_index,
    index_fingerprint,
    load_or_build_index,
    reset_index,
)

__all__ = [
    "CHUNK_CHARS",
    "DEFAULT_ARTICLES_DIR",
    "DEFAULT_INDEX_DIR",
    "EMBEDDING_MODEL",
    "OVERLAP_CHARS",
    "Article",
    "ArticleFormatError",
    "Chunk",
    "Embedder",
    "Hit",
    "IndexFormatError",
    "KnowledgeIndex",
    "SentenceTransformerEmbedder",
    "articles_fingerprint",
    "build_index",
    "chunk_article",
    "chunk_articles",
    "get_index",
    "index_fingerprint",
    "load_articles",
    "load_or_build_index",
    "parse_article",
    "reset_index",
    "source_name",
]
