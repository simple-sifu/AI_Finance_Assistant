---
title: 'Docker image and AWS deployment'
type: 'feature'
created: '2026-10-09'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
baseline_commit: '35d955fa351c2ec3daef36b8fd0d2cad27496cbf'
context:
  - '{project-root}/_bmad-output/specs/spec-ai-finance-assistant/SPEC.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** CAP-11 needs a grader to open a public URL and finish the demo against the deployed container. Today the app only runs locally.

**Success signal:** in a live demo with the Alpha Vantage quota exhausted, one question per agent is routed correctly and answered.

**Approach:**
- Build one Docker image:
  - CPU-only torch;
  - the FAISS index built at image build time;
  - the embedding model cached in the image and used offline at runtime.
- Run it as a long-running container on one AWS EC2 instance, from a runbook the human follows.

**Decisions (human, 2026-10-09):**
- **Hosting:** EC2 `t3.small` (2 GB) on a new AWS free-plan account, so usage comes out of the credits. Ubuntu 24.04, 20 GB disk, a 2 GB swap file for the build, and an Elastic IP.
- **Deploy:** this session writes the Dockerfile, a server setup script and a step-by-step console runbook. The human creates the instance and runs the commands over SSH, and the image builds on the server. Docker and the AWS CLI are not used on the Mac.
- **Access:** a shared password screen shows before the tabs when `APP_PASSWORD` is set in the server env file. Without it (local dev) there is no screen.
- **Address:** plain HTTP at `http://<elastic-ip>` on port 80. No domain, no TLS.

## Boundaries & Constraints

**Always:**
- **Image build:**
  - CPU-only torch wheels.
  - The index is built during `docker build`, and the build fails if it can't be built.
  - The build checks that the index loads and answers a search with `HF_HUB_OFFLINE=1`.
- **Secrets:** API keys and `APP_PASSWORD` reach the container only at run time, from `--env-file` on the server. They are never in the image or the repo.
- **App health:**
  - Without keys, the container starts and every tab renders.
  - `/_stcore/health` is the Docker healthcheck.
  - `--restart unless-stopped` brings the app back after a crash or reboot.
- **Password screen:** a wrong password shows an error and nothing else; no agent is reachable before the password is correct. The comparison is constant-time, and the password is never logged.
- **Runbook:**
  - The security group opens only SSH (from "My IP") and HTTP 80.
  - It explains how to stop or terminate the instance when grading is over.

**Never:**
- Changing agent, router, guardrail or market-data behavior. The only UI change is the password screen.
- A separate frontend or backend.
- Serverless hosting.
- Committing `.env`, keys or AWS credentials.
- Creating AWS resources from this session.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Cold start offline | Container with no network to Hugging Face | Finance Q&A answers from the built index and cached model | `HF_HUB_OFFLINE=1`, no download |
| No keys | Container run without keys | All five tabs render; questions show the missing-key message | Story 9 behavior |
| Quota exhausted | `MARKET_DATA_MODE=mock`, or Alpha Vantage's rate-limit reply | Market answer from cache or mock data, labelled as such | Story 1 fallback |
| Password set | `APP_PASSWORD` set, visitor not signed in | Only a password field; no tabs | N/A |
| Wrong password | Incorrect entry | "Incorrect password" message, still no tabs | N/A |
| Right password | Correct entry | Tabs show for the rest of the browser session | N/A |
| No password configured | `APP_PASSWORD` unset | Tabs show directly | N/A |
| Crash or reboot | Process exits or the server restarts | Container comes back on its own | `--restart unless-stopped` |

</frozen-after-approval>

## Code Map

- `pyproject.toml`, `uv.lock`: `[tool.uv.sources]` already routes torch to `https://download.pytorch.org/whl/cpu` on Linux, and the lock includes those wheels. In the image, `uv sync --frozen --no-dev` installs the project editable from `/app`.
- `src/finance_assistant/knowledge/articles.py:12`:
  - `PROJECT_ROOT = Path(__file__).resolve().parents[3]` is `/app` with that install.
  - `DEFAULT_INDEX_DIR` is under it.
  - Closes the story 3 path item without a code change.
- `src/finance_assistant/knowledge/index.py`: `EMBEDDING_MODEL`, `get_index()`, `KnowledgeIndex.search`. Use them for the offline check during the build.
- `scripts/build_index.py`: run during `docker build`. It downloads the model into `HF_HOME`, which closes the story 3 Hugging Face item.
- `src/finance_assistant/config.py`:
  - `Settings` and `load_settings` (env vars plus an optional `.env` in the working directory).
  - Add `app_password`, redacted in `__repr__`, read from `APP_PASSWORD`.
  - The container's working directory has no `.env`, so only `--env-file` applies.
- `src/finance_assistant/ui/app.py`: `main()` (set_page_config, title, `_install_agents`, tabs). The password screen runs after the title and before `_install_agents`/tabs; `st.stop()` until `st.session_state` marks the session signed in.
- `tests/ui/test_app.py`: AppTest fixtures (`AppTest.from_file`, `FakeClassifier`, `RecordingAgent`). Reuse them for the password tests.
- `.streamlit/config.toml`: copy into the image.
- `.env.example` and `README.md`: document `APP_PASSWORD`.
- `_bmad-output/implementation-artifacts/deferred-work.md`: close the two story 3 items (Hugging Face cache and articles path).

## Tasks & Acceptance

**Execution:**
- [x] `src/finance_assistant/config.py`, `.env.example`: add `APP_PASSWORD` to `Settings`, redacted. Unit-test the load and the redaction.
- [x] `src/finance_assistant/ui/app.py`: add the password screen per the matrix. AppTest cases for set, wrong, right and unset.
- [x] `Dockerfile`, `.dockerignore`:
  - `python:3.12-slim` with uv.
  - Install dependencies first (layer cache), then copy the code and articles.
  - Build the index with `HF_HOME` inside the image, set `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`, and run a search check during the build.
  - Run as a non-root user, `EXPOSE 8501`, `HEALTHCHECK` on `/_stcore/health`.
  - `CMD` is `streamlit run app.py` headless on `0.0.0.0:8501`.
  - `.dockerignore` excludes `.env`, `.venv`, `.git`, `tests`, `_bmad*`, `.claude`, `.agents` and `knowledge_base/index`.
- [x] `deploy/setup-server.sh`: idempotent, for Ubuntu 24.04.
  - Add 2 GB swap if missing.
  - Install Docker.
  - Clone or pull the public repo.
  - `docker build`.
  - Replace the running container (`--env-file ~/finance-assistant.env`, `-p 80:8501`, `--restart unless-stopped`).
  - Wait for the health check and print the URL.
- [x] `deploy/RUNBOOK.md`: console steps.
  - Billing alert or budget; key pair.
  - Launch `t3.small` Ubuntu 24.04 with 20 GB gp3 and a security group (SSH from My IP, HTTP 80).
  - Elastic IP; SSH command; env file template; running the script.
  - Verify: health check, one question per agent, `MARKET_DATA_MODE=mock` as the quota-exhausted check, offline cold start.
  - How to update after a merge; how to stop and terminate.
- [x] `README.md`: deployment section linking the runbook. Close the story 3 deferred items.

**Acceptance Criteria:**
- Given the offline suite, when `uv run pytest -q` runs, then all tests pass, including the password tests.
- Given an Ubuntu 24.04 `t3.small`, when the runbook is followed, then `http://<elastic-ip>` asks for the password and, after it, completes one question per agent with the disclaimer.

## Implementation Notes

- Implemented directly in this session (no implementation subagent).
- **Password screen:** `ui/app.py` `_signed_in()` runs before `_install_agents()` and the tabs.
  - It uses a form (`password_form`, key `password`), compares with `hmac.compare_digest`, sets `st.session_state["signed_in"]`, and calls `st.stop()` until signed in.
  - `APP_PASSWORD` is stripped and reads as None when blank, redacted in `Settings.__repr__`, and added to the test env isolation list.
- **Dockerfile:**
  - uv 0.12.3 binary copied from `ghcr.io/astral-sh/uv`.
  - Dependencies install in their own layer.
  - `README.md` is copied because hatchling reads it as package metadata.
  - The app runs as a non-root `app` user with no `chown -R` (that would duplicate the torch layer); the app only reads `/app`.
  - If an index save ever fails at runtime, `load_or_build_index` already falls back to memory.
- **`deploy/setup-server.sh`:**
  - Uses Ubuntu's `docker.io` package and `sudo docker`, so there's no docker-group relogin.
  - Fast-forwards only (`merge --ff-only`), so a hand-edited server checkout stops the script instead of being overwritten.
  - Stops after creating the env file the first time.
  - Warns when `APP_PASSWORD` or `OPENAI_API_KEY` is empty.
  - Reads the public IP from IMDSv2.
- **Matrix coverage:**
  - Password set, wrong, right and unset, plus no keys: AppTest, offline.
  - Quota exhausted: story 1 and story 5 fallback tests (unchanged), plus runbook step 7.4.
  - Cold start offline: the build-time `RUN` search with `HF_HUB_OFFLINE=1`, plus runbook step 7.5.
  - Crash or reboot: `--restart unless-stopped`.
  - The Docker build and the restart behavior can't run here (no Docker on this Mac). The human verifies them on the server per the runbook.
- 687 offline tests pass; `bash -n deploy/setup-server.sh` is clean.
- Post-review (2026-10-09): fixes for triage rows 1–6, 8, 9 and 14–18 applied.
  - `deploy/check_index.py` passes locally with `HF_HUB_OFFLINE=1` (76 articles).
  - 687 offline tests pass; the script syntax check is clean.
  - Not runnable here: `docker build` and the server run. The human verifies them per the runbook.
- Known gap against the frozen matrix wording: sign-in lasts for the browser tab's session, not the whole browser. A refresh asks again (Streamlit session state). It is documented in the runbook and reported to the human.

## Spec Change Log

## Review Triage Log

Pass 1 (2026-10-09): layers blind (B), edge-case (E), verification-gap (V; no gaps, one other finding).

| # | Layer | Finding | Verdict | Evidence | Route |
|---|-------|---------|---------|----------|-------|
| 1 | B, E, V | Script's empty-value checks miss whitespace-only `APP_PASSWORD`/`OPENAI_API_KEY`; app strips them to None, so it goes public with no warning | medium | `grep '^APP_PASSWORD=.+'` vs `load_settings` `.strip() or None` | patch (`has_value` regex) |
| 2 | B | Empty `APP_PASSWORD` only warns; spec says the deployed app sits behind the password | medium | Script continued after the warning | patch (script exits with an error) |
| 3 | B, E | Build-time offline check uses `get_index()`, which quietly rebuilds in memory if the saved index fails to load | medium | `load_or_build_index` catches `IndexFormatError` and rebuilds | patch (`deploy/check_index.py` calls `KnowledgeIndex.load` directly) |
| 4 | B, E | Offline check runs as root, app runs as `app` | low | `RUN` check came before `USER app` | patch (check runs after `USER app`) |
| 5 | B, E | Entered password not stripped while the stored one is | low | `compare_digest(entered.encode(), …)` | patch (+ test) |
| 6 | E | Plaintext password stays in widget state after sign-in | low | Form had no `clear_on_submit` | patch (`clear_on_submit=True`) |
| 7 | E | Unlimited password attempts | low | Shared demo password over HTTP (human decision); a lockout adds state and branches | rejected (runbook security note) |
| 8 | B | Server's downloaded copy of the script never updates; running the repo copy while `git pull` rewrites it could break bash | medium | Runbook ran `~/setup-server.sh` every time | patch (runbook runs the repo copy; body wrapped in `main()`) |
| 9 | B | Runbook curl URL needs a merged `main` | low | Script exists only after merge | patch (doc note) |
| 10 | B, E | `docker run` failure after `rm --force` leaves no app; no rollback | low | Rare (port conflict); re-running fixes it; rollback adds image tagging and branches | rejected |
| 11 | E | `APP_DIR` exists without `.git` after an interrupted clone | low | Unlikely; a clear git error results | rejected |
| 12 | E | Stale `/swapfile` with swap off | low | Unlikely on a fresh instance | rejected |
| 13 | E | Unhealthy (hung) container not restarted | low | `--restart` covers exits only; a hang is rare; the runbook has `docker restart` | rejected |
| 14 | B | Build cache fills the 20 GB disk over rebuilds | low | `image prune` removes dangling images only | patch (`builder prune --filter until=168h`) |
| 15 | B | `docker.io` without `docker-buildx` uses the deprecated legacy builder | low | Ubuntu 24.04 packages them separately | patch |
| 16 | B | No cleartext/brute-force warning; sign-in is per tab, not per browser session | low | HTTP decision; `st.session_state` is per websocket session | patch (runbook security notes); the matrix wording "rest of the browser session" is reported to the human |
| 17 | B | Env-file duplicate keys, inline comments | low | Template is copied from `.env.example` | patch (doc) |
| 18 | B | Hard-coded "76 articles" in the runbook | low | Drifts with the knowledge base | patch |
| 19 | B | Password tests miss whitespace-only `APP_PASSWORD` and the `_install_agents`-before-sign-in check | low | Whitespace is covered by the config test (→ None → no screen); `st.stop()` precedes `_install_agents` and the test shows no agent requests | rejected |
| 20 | B | Articles-path closure is unguarded | false | A non-editable install would leave `/app/knowledge_base/articles` unfound and `build_index` raises "no knowledge-base articles found", failing `docker build` | rejected |
| 21 | E | `HF_HOME` unwritable for `app` at runtime | maybe-false | Settled by #4: the check now runs as `app` offline, so the build fails if loading needs writes | patch (#4) |

## Verification

**Commands:**
- `uv run pytest -q`: expected, all pass offline.
- `bash -n deploy/setup-server.sh`: expected, no syntax errors.

**Manual checks:**
- On the server: the `docker build` log shows the index built and the offline search check passed.
- `curl -s localhost/_stcore/health` returns `ok`.
- Browser at `http://<elastic-ip>`: password, then one question per agent (with `MARKET_DATA_MODE=mock` for the quota-exhausted run).
