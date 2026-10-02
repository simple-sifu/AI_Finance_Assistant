---
title: 'Knowledge base, FAISS index and Finance Q&A agent'
type: 'feature'
created: '2026-10-01'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
baseline_commit: '51ae62ea99178e7fbb61f94f0a07c007656117c1'
context:
  - '{project-root}/_bmad-output/specs/spec-ai-finance-assistant/SPEC.md'
  - '{project-root}/_bmad-output/specs/spec-ai-finance-assistant/stack.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The `finance_qa` route still returns a stub. CAP-2 needs beginner explanations grounded in the curated knowledge base, with the articles cited. The 76 articles exist (`knowledge_base/articles/`, commit 7f1bd63), but nothing indexes or retrieves them.

**Approach:** Chunk the articles, embed them with `all-MiniLM-L6-v2`, and store them in a FAISS index built ahead of time by a script (later run at Docker build time, per stack.md). A real Finance Q&A agent retrieves the top chunks for a question and has the LLM answer from them under `build_system_prompt`, returning the cited articles as `AgentResult.sources`. Its replies pass through the existing advice review like any real agent's.

## Boundaries & Constraints

**Always:**
- Answers come from retrieved article text; every source returned is an article the answer actually drew on, with its title and URL.
- The system prompt is built with `build_system_prompt`; the agent never adds the disclaimer.
- The embedding model and index load once per process and lazily (first question), never at import.
- Default tests stay offline: an injectable embedder and the existing respx OpenAI mocks; no model download, no network.
- LLM calls use `chat_model` per call; logs carry exception types only.

**Never:** Re-fetching or editing articles (the builder script is out of scope). Vector stores other than FAISS. Committing the built index. Changing the router, guardrail or review behavior.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Grounded answer | "What is compound interest?" | Plain-language answer drawn from the compound-interest article; `sources` includes it (title + URL) | N/A |
| Multi-article | "ETFs vs mutual funds?" | Answer cites each article it used; no uncited sources | N/A |
| Advice-seeking | "Which index fund should I buy?" (`seeks_advice=True`) | Opens with the redirect, then educates from articles; passes advice review with sources kept | N/A |
| Nothing relevant | Off-KB finance question (best chunk below threshold) | "My articles don't cover that yet" plus the topics they do cover; no LLM answer, `sources=[]` | N/A |
| Index missing | No built index on disk | Built on first use from the articles, then reused | Info log once |
| LLM failure | OpenAI error during answer | Agent raises; graph gives `AGENT_FAILURE_TEXT` (existing behavior) | Existing warning |

**Decisions (human, 2026-10-01):** Index all 76 articles; citations show source name and URL, which meets CC BY-SA attribution for the 10 Wikipedia articles (this answers the SPEC's article-sourcing open question). When nothing is relevant, say the articles don't cover it, and never answer without a source. Citations are inline `[n]` markers matching the order of `sources`, and only cited articles are returned. The spec is kept whole at ~1,800 tokens; the size risk was accepted.

</frozen-after-approval>

## Code Map

- `knowledge_base/articles/*.md` -- YAML front matter (`title`, `source`, `url`, `license`, `category`, `retrieved`) and a Markdown body; 642 B–~20 kB each. Read-only here.
- `src/finance_assistant/knowledge/` (new) -- `articles.py` (load and parse front matter), `chunking.py` (split by heading/paragraph, ~800 chars with overlap; each chunk keeps its article's title/url), `index.py` (embedder protocol, `SentenceTransformerEmbedder`, FAISS build/save/load, `search(query, k) -> list[Hit]` with scores), and a lazy process-wide accessor with `reset_*()` for tests (mirror `market_data/client.py:get_client`/`reset_client`).
- `scripts/build_index.py` (new) -- builds the index into `knowledge_base/index/` (gitignored).
- `src/finance_assistant/tutor/finance_qa.py` (new) -- `FinanceQAAgent.run(request)`: retrieve (question, plus the previous user turn for short follow-ups), prompt with numbered excerpts, answer, and map the cited articles to `Source`s. Reuse `tutor/llm.py:chat_model`, `guardrail.build_system_prompt`, `models.AgentResult/Source`.
- `src/finance_assistant/tutor/agents.py` -- add `install_real_agents()`, which registers `FinanceQAAgent` for `finance_qa`. Do not touch `_STUBS`/`reset_agents`, so existing tests keep stubs. The app (story 9) calls it.
- `pyproject.toml` -- add `faiss-cpu` and `sentence-transformers`. Route `torch` to the PyTorch CPU index on Linux via `[tool.uv.sources]`. Add `knowledge_base/index/` to `.gitignore`.
- `tests/tutor/test_ask.py:101` (`test_each_agent_route`) -- relies on the stub for `finance_qa`; it must stay green because `install_real_agents` is opt-in.

## Tasks & Acceptance

**Execution:**
- [x] `pyproject.toml`, `.gitignore`, `uv.lock` -- dependencies, CPU torch on Linux, ignore the built index.
- [x] `src/finance_assistant/knowledge/*`, `scripts/build_index.py` -- load, chunk, embed, build/save/load, search, lazy accessor.
- [x] `src/finance_assistant/tutor/finance_qa.py`, `agents.py`, `__init__.py` -- agent, inline `[n]` citation mapping, the no-match reply, `install_real_agents()`.
- [x] `tests/knowledge/test_*.py` -- parsing all 76 real articles, chunk boundaries and metadata, round-trip save/load, and search ranking with a deterministic fake embedder.
- [x] `tests/tutor/test_finance_qa.py` -- every matrix row offline (fake embedder, respx OpenAI); through `ask` with `install_real_agents()` and a fake reviewer, sources survive review.
- [x] `tests/tutor/test_finance_qa_live.py` -- `@pytest.mark.live`: 5 CAP-2 questions on the real index, model and API; each answer cites ≥1 expected article and passes the real advice reviewer.

**Acceptance Criteria:**
- Given a fresh clone, when `uv run python scripts/build_index.py` runs, then the index is written under `knowledge_base/index/` covering all 76 articles.
- Given the default test run, when `uv run pytest` runs, then everything passes offline and nothing downloads a model.
- Given `install_real_agents()` and a real key, when `ask("What is compound interest?")` runs, then the reply cites the compound-interest article and carries the disclaimer once.

## Implementation Notes

- **Index:** 76 articles -> 527 chunks (heading/paragraph split, 800 chars, 150-char overlap; headings kept as chunk metadata and prefixed to the embedded text). `IndexFlatIP` over L2-normalized vectors, so scores are cosine similarity. `chunks.json` stores the embedding model name and a fingerprint of the article files plus `CHUNK_CHARS`/`OVERLAP_CHARS`; a saved index built for other articles or another model is rebuilt on first use (info log once) and saved if the directory is writable, otherwise kept in memory. Files are written atomically with mode 0644.
- **Relevance threshold:** `MIN_SCORE = 0.40`, calibrated on the real index: on-KB questions top out at 0.43-0.88; off-KB finance questions (mortgage, crypto staking, filing taxes, options greeks) at 0.30-0.38. "What is a credit score?" scores 0.45 against an unrelated chunk, so a second gate exists: the model replies exactly `NOT_COVERED` (whole-reply match) when the excerpts don't answer, and an answer with no valid `[n]` citation is also replaced by the not-covered reply. Neither path returns sources.
- **Citations:** articles are numbered in the prompt (up to 4 articles, 3 chunks each, from the top 8 chunks); after the answer, cited numbers are renumbered by first use, invalid numbers are dropped, `[1, 2]`, `[1; 2]` and ranges like `[1-3]`/`[1–3]` are expanded to `[1][2]...`, and only cited articles become `Source(title="<title> (<publisher>)", url=...)`, e.g. "Compound interest (Wikipedia)".
- **Advice:** when `seeks_advice`, the prompt asks the model to open with `ADVICE_REDIRECT`; the agent prepends it if missing. A no-match reply to an advice question is also prefixed with the redirect.
- **Follow-ups:** the bare question is always searched; a question of 6 words or fewer is also searched as "question + previous user turn". Hits are merged by rank (bare first), keeping each chunk's best score, so a short new-topic question still finds its own article. The last 4 history turns (clipped) go into the prompt as context.
- **macOS OpenMP clash:** the faiss-cpu and torch macOS wheels each bundle libomp; using both in one process aborts (`OMP: Error #15`) or segfaults. `knowledge.index._faiss()` imports faiss lazily and on darwin only sets `KMP_DUPLICATE_LIB_OK=TRUE`, imports torch before faiss, and sets faiss to one thread. This was verified 10/10 in a stress script plus the live eval. Linux (Docker) wheels share libgomp and skip the workaround.
- **Dependencies:** `faiss-cpu`, `sentence-transformers`, `numpy`, `pyyaml`, and `torch` as a direct dependency, because `[tool.uv.sources]` only routes direct dependencies to the `pytorch-cpu` index (Linux marker, `explicit = true`).
- **Test isolation:** the root `conftest.py` points `DEFAULT_INDEX_DIR` at a temp dir and makes `SentenceTransformerEmbedder._load` raise, so default tests can neither overwrite the developer's built index nor load or download the model. Tests use `tests/knowledge/fakes.py:HashEmbedder` (hashed bag of words).
- Post-review verification (2026-10-01): 274 offline tests pass; `build_index.py` indexed 76 articles as 527 chunks (files mode 0644); live test 8/8 on gpt-4o-mini, including both off-topic questions returning no sources, after the review patches.

## Spec Change Log

## Review Triage Log

Pass 1 (2026-10-01): layers blind (B), edge-case (E), verification-gap (V). The review diff left out `uv.lock`; the triager checked it directly.

| # | Layer | Finding | Verdict | Evidence | Route |
|---|-------|---------|---------|----------|-------|
| 1 | B, E | Short new-topic questions (≤6 words) are joined to the previous turn, so retrieval drifts and the relevance gate weakens | medium | `retrieval_query` (finance_qa.py:95) concatenates for any ≤6-word question, e.g. "What is the rule of 72?" | patch |
| 2 | E | A long previous turn placed before the follow-up can push the follow-up past the embedder's input limit | low | Same root as #1 (question placed after the previous turn) | patch (with #1) |
| 3 | B, E | Range or odd citation markers (`[1-3]`, `[1–2]`, `[1; 2]`) keep stale numbers after renumbering | medium | `_CITATION_GROUP_RE` matches only `[n]`/`[n, m]` | patch |
| 4 | E | Bracketed non-citations like `[2020]` are deleted | low | Out-of-range numbers are dropped by design; years in brackets are rare in answers | rejected |
| 5 | B | `NOT_COVERED` is matched as a substring | low | A real answer mentioning the sentinel is withheld; the fix is a whole-reply compare | patch |
| 6 | B | List-form `message.content` becomes "" with no log | low | Possible with reasoning models (gpt-5 is allowed in stack.md); the fix is a join | patch |
| 7 | B | Chunking parameters are not in the index fingerprint | low | A change to `CHUNK_CHARS`/`OVERLAP_CHARS` loads a stale index unless `FORMAT_VERSION` is bumped | patch |
| 8 | B, V | Offline "nothing relevant" tests depend on hash-embedder word overlap | low | V: these tests pin the gate only in `HashEmbedder` score space | patch |
| 9 | B, E | `TOPICS` can drift from article categories; the `retirement-and-tax` label omits tax | low | No test compares `TOPICS` with the loaded categories | patch |
| 10 | B | `test_llm_failure_raises_from_the_agent` uses `pytest.raises(Exception)` | low | Any bug passes it | patch |
| 11 | B, V | `scripts/build_index.py` is never run by a test; no `--articles` option | medium | V: nothing imports the script, and it is the first acceptance criterion | patch (test only) |
| 12 | B | `uv.lock` not reviewable | false | Triager read `uv.lock`: Linux resolves `torch 2.14.1+cpu` from download.pytorch.org/whl/cpu; macOS from PyPI | rejected |
| 13 | B | Model download on the first question in Docker; HF cache not kept; no warm-up | maybe-false | Depends on the Dockerfile (story 11) | defer (medium, unverified) |
| 14 | B | A cited-nothing answer is reported as "not covered" | low | By design: the human decided never to answer without a source | rejected |
| 15 | B | Advice test asks "…buy? should i" with no explanation | low | `test_finance_qa.py:248`; use the clean matrix question | patch |
| 16 | B | `install_real_agents()` → `reset_agents()` round trip untested | low | The opt-in design relies on it | patch |
| 17 | B, E | Front matter split on the substring `"\n---"` | low | The articles are generated, but the fix is a direct regex change | patch |
| 18 | V | `group_hits` caps (4 articles, 3 chunks) unpinned | medium | Pre-verified: the caps set to 100 pass all tests | patch |
| 19 | V | History block in the prompt unchecked | medium | Pre-verified: dropping history passes all tests | patch |
| 20 | V | Index file mode 0644 unverified | low | Pre-verified: a no-op `chmod` passes | patch |
| 21 | V | Relevance threshold never checked live on off-topic questions | medium | The live eval has only on-topic cases | patch |
| 22 | E | A failed index load retries the full build on every question while holding the lock | low | Retrying after a transient download failure is reasonable; no named harm at this traffic | rejected |
| 23 | E | A non-editable install breaks `parents[3]` KB path resolution | maybe-false | Only if story 11 installs the package non-editable | defer (medium, unverified) |
| 24 | E | `chunk_chars <= 0` loops forever | low | No caller passes it; constants only | rejected |
| 25 | E | `#` lines in code fences become headings | low | No article contains code fences | rejected |
| 26 | E | faiss imported elsewhere first skips the macOS workaround | low | Only `knowledge.index` imports faiss | rejected |
| 27 | E | Index dimension mismatch under the same model name | false | The same model name gives the same dimension | rejected |
| 28 | E | A non-OSError during save discards the built index | low | Unlikely in normal use; fix adds a branch | rejected |
| 29 | E | NaN scores pass the threshold | low | Embeddings are normalized; NaN needs a broken model | rejected |
| 30 | E | A quoted or reworded redirect gets duplicated | low | Cosmetic; fix adds normalization logic | rejected |
| 31 | E | The canned not-covered reply goes through advice review (extra call; fails closed on outage) | low | The fix changes review behavior, which the frozen intent excludes ("Never: changing … review behavior") | rejected |
| 32 | E | Non-UTF-8 article gives an untyped error | low | Articles are generated UTF-8 | rejected |
| 33 | V | macOS OpenMP workaround not exercised offline | low | Matches the design; covered by the live eval only | rejected (surfaced to human) |

## Design Notes

`install_real_agents()` keeps real agents opt-in. Making the registry default to real agents would send every existing offline test that routes to `finance_qa` through the model and OpenAI. The app turns real agents on explicitly, and stories 4–8 add theirs to the same function.

## Verification

**Commands:**
- `uv run pytest -q` -- expected: all pass offline, live deselected
- `uv run python scripts/build_index.py` -- expected: index written, chunk count printed
- `uv run pytest -q -m live tests/tutor/test_finance_qa_live.py` -- expected: 5/5 cited and reviewed
