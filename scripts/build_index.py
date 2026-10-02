"""Build the FAISS knowledge index from knowledge_base/articles into knowledge_base/index/.

Run ahead of time (e.g. at Docker image build) so the app only embeds queries:

    uv run python scripts/build_index.py

The first run downloads the all-MiniLM-L6-v2 embedding model (~90 MB).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from finance_assistant.knowledge import DEFAULT_INDEX_DIR, build_index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_INDEX_DIR, help="output directory")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    index = build_index()
    index.save(args.out)
    print(f"Indexed {index.article_count} articles as {len(index.chunks)} chunks into {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
