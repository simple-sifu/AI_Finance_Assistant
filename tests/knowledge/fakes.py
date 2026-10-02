"""A deterministic, offline stand-in for the sentence-transformers embedder."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i in is it its my of on or so that the their "
    "this to vs was what when which who why will with you your".split()
)


def _token_id(token: str, dim: int) -> int:
    return int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "little") % dim


class HashEmbedder:
    """Bag-of-words hashed into ``dim`` buckets: texts sharing words score higher. Counts calls."""

    name = "test-hash-embedder"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.calls: list[int] = []  # batch size of each embed call

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append(len(texts))
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                if token not in _STOPWORDS:
                    token = token[:-1] if len(token) > 3 and token.endswith("s") else token
                    out[row, _token_id(token, self.dim)] += 1.0
        return out
