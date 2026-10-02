"""Embed chunks, build/save/load a FAISS index, and search it.

Vectors are L2-normalized and stored in an inner-product index, so a hit's
score is its cosine similarity to the query (higher is closer, at most 1).

The process-wide index (``get_index``) is created lazily on the first search:
it loads ``knowledge_base/index/`` if that was built for the current articles
and embedding model, and otherwise builds it from the articles and tries to
save it. Nothing heavy (torch, the model, the index) loads at import.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import threading
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import faiss

import numpy as np

from .articles import PROJECT_ROOT, articles_fingerprint, load_articles
from . import chunking
from .chunking import Chunk, chunk_articles

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "knowledge_base" / "index"
INDEX_FILE = "index.faiss"
META_FILE = "chunks.json"
FORMAT_VERSION = 1


def _faiss():
    """Import faiss lazily, working around a macOS OpenMP clash with torch.

    On macOS the faiss-cpu and torch wheels each bundle their own libomp. Two
    OpenMP runtimes in one process abort ("OMP: Error #15") or segfault once both
    run parallel regions. The combination that works: allow the duplicate
    runtime, load torch before faiss, and keep faiss single-threaded (a flat
    index over a few hundred chunks doesn't need threads). Linux wheels share
    libgomp and are unaffected, so the Docker image skips all of this.
    """
    if sys.platform == "darwin" and "faiss" not in sys.modules:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        try:
            import torch  # noqa: F401 - must load before faiss
        except ImportError:  # pragma: no cover - torch is a dependency
            pass
        import faiss

        faiss.omp_set_num_threads(1)
        return faiss
    import faiss

    return faiss


@runtime_checkable
class Embedder(Protocol):
    """Turns texts into vectors. ``name`` identifies the model a saved index was built with."""

    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """``all-MiniLM-L6-v2`` via sentence-transformers; the model loads on first ``embed``."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                _faiss()  # on macOS, sets up the torch/faiss OpenMP workaround first
                from sentence_transformers import SentenceTransformer  # heavy: torch

                logger.info("Loading embedding model %s", self.name)
                self._model = SentenceTransformer(self.name, device="cpu")
            return self._model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        vectors = model.encode(
            list(texts), batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype="float32")


class IndexFormatError(ValueError):
    """A saved index is missing, corrupt, or was built for other articles or another model."""


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def _normalized(vectors: np.ndarray) -> np.ndarray:
    vectors = np.ascontiguousarray(np.asarray(vectors, dtype="float32"))
    if vectors.ndim != 2:
        raise ValueError("embedder must return a 2-D array")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class KnowledgeIndex:
    """FAISS inner-product index over chunks, plus the embedder for queries."""

    def __init__(self, index: faiss.Index, chunks: list[Chunk], embedder: Embedder, fingerprint: str = "") -> None:
        if index.ntotal != len(chunks):
            raise IndexFormatError("index and chunk metadata disagree on size")
        self._index = index
        self.chunks = chunks
        self.embedder = embedder
        self.fingerprint = fingerprint
        self._search_lock = threading.Lock()

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder, fingerprint: str = "") -> KnowledgeIndex:
        if not chunks:
            raise ValueError("cannot build an index with no chunks")
        vectors = _normalized(embedder.embed([c.embed_text for c in chunks]))
        if vectors.shape[0] != len(chunks):
            raise ValueError("embedder returned the wrong number of vectors")
        index = _faiss().IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index, chunks, embedder, fingerprint)

    @property
    def article_count(self) -> int:
        return len({c.article_slug for c in self.chunks})

    def search(self, query: str, k: int = 5) -> list[Hit]:
        """The ``k`` chunks closest to ``query``, best first."""
        if not query.strip() or k <= 0:
            return []
        vector = _normalized(self.embedder.embed([query]))
        with self._search_lock:
            scores, ids = self._index.search(vector, min(k, len(self.chunks)))
        return [Hit(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0], strict=True) if i >= 0]

    def save(self, directory: Path) -> None:
        """Write ``index.faiss`` and ``chunks.json`` into ``directory`` (each replaced atomically)."""
        directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "format": FORMAT_VERSION,
            "model": self.embedder.name,
            "fingerprint": self.fingerprint,
            "chunks": [asdict(c) for c in self.chunks],
        }
        _atomic_write(directory / INDEX_FILE, _faiss().serialize_index(self._index).tobytes())
        _atomic_write(directory / META_FILE, json.dumps(meta, ensure_ascii=False).encode("utf-8"))

    @classmethod
    def load(cls, directory: Path, embedder: Embedder, fingerprint: str | None = None) -> KnowledgeIndex:
        """Load a saved index; raise IndexFormatError if it is unusable with ``embedder``/``fingerprint``."""
        try:
            meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
            raw = np.frombuffer((directory / INDEX_FILE).read_bytes(), dtype="uint8")
        except (OSError, ValueError) as exc:
            raise IndexFormatError(f"cannot read saved index ({type(exc).__name__})") from exc
        if not isinstance(meta, dict) or meta.get("format") != FORMAT_VERSION:
            raise IndexFormatError("saved index has an unknown format")
        if meta.get("model") != embedder.name:
            raise IndexFormatError("saved index was built with a different embedding model")
        if fingerprint is not None and meta.get("fingerprint") != fingerprint:
            raise IndexFormatError("saved index was built from different articles")
        try:
            chunks = [Chunk(**c) for c in meta["chunks"]]
            index = _faiss().deserialize_index(raw.copy())
        except Exception as exc:  # noqa: BLE001 - any corruption means "rebuild"
            raise IndexFormatError(f"saved index is corrupt ({type(exc).__name__})") from exc
        return cls(index, chunks, embedder, str(meta.get("fingerprint") or ""))


def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.chmod(tmp, 0o644)  # mkstemp makes 0600; the app may run as another user than the builder
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def index_fingerprint(articles_dir: Path | None = None) -> str:
    """Identifies what a built index depends on: the article files and the chunking settings."""
    settings = f"chunk_chars={chunking.CHUNK_CHARS};overlap_chars={chunking.OVERLAP_CHARS}"
    return f"{articles_fingerprint(articles_dir)}:{settings}"


def build_index(
    embedder: Embedder | None = None, articles_dir: Path | None = None
) -> KnowledgeIndex:
    """Load, chunk, and embed every article into a new in-memory index."""
    embedder = embedder if embedder is not None else SentenceTransformerEmbedder()
    articles = load_articles(articles_dir)
    if not articles:
        raise ValueError("no knowledge-base articles found")
    return KnowledgeIndex.build(chunk_articles(articles), embedder, index_fingerprint(articles_dir))


def load_or_build_index(
    embedder: Embedder | None = None,
    index_dir: Path | None = None,
    articles_dir: Path | None = None,
) -> KnowledgeIndex:
    """Load the saved index if it matches the articles and model; otherwise build it and try to save it."""
    embedder = embedder if embedder is not None else SentenceTransformerEmbedder()
    index_dir = index_dir if index_dir is not None else DEFAULT_INDEX_DIR
    fingerprint = index_fingerprint(articles_dir)
    try:
        return KnowledgeIndex.load(index_dir, embedder, fingerprint)
    except IndexFormatError as exc:
        logger.info("No usable knowledge index at %s (%s); building it from the articles", index_dir, exc)
    index = build_index(embedder, articles_dir)
    try:
        index.save(index_dir)
    except OSError as exc:
        logger.warning("Could not save the knowledge index (%s); keeping it in memory", type(exc).__name__)
    return index


_default_index: KnowledgeIndex | None = None
_default_index_lock = threading.Lock()


def get_index(
    *,
    embedder: Embedder | None = None,
    index_dir: Path | None = None,
    articles_dir: Path | None = None,
) -> KnowledgeIndex:
    """Return the process-wide index, loading or building it on first use.

    The arguments only matter on the call that creates it (tests pass a fake embedder).
    """
    global _default_index
    with _default_index_lock:
        if _default_index is None:
            _default_index = load_or_build_index(embedder, index_dir, articles_dir)
        return _default_index


def reset_index() -> None:
    """Drop the process-wide index (and its embedder); mainly for tests."""
    global _default_index
    with _default_index_lock:
        _default_index = None
