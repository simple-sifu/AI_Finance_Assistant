# AI Finance Tutor: one Streamlit container (story 11). Build on the server with
#   docker build -t finance-assistant .
# API keys and APP_PASSWORD come at run time from an env file mounted read-only at
# /app/.env, which the app parses like local dev; none are baked in.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Dependencies first, so code edits don't reinstall torch. On Linux the lock
# resolves torch to the CPU-only wheels (pyproject [tool.uv.sources]).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# The project itself, installed editable from /app so knowledge/articles.py finds
# /app/knowledge_base (PROJECT_ROOT = parents[3]).
COPY README.md app.py ./
COPY .streamlit ./.streamlit
COPY src ./src
COPY scripts ./scripts
COPY knowledge_base/articles ./knowledge_base/articles
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# Build the FAISS index now; this also downloads the embedding model into HF_HOME.
# A failure here (e.g. no articles) fails the build.
RUN python scripts/build_index.py

# From here on, never reach Hugging Face: the model must come from the image.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Run unprivileged. /app stays root-owned and read-only to the app: it only reads
# the code, index and model (no chown, which would copy the torch layer).
RUN useradd --create-home --uid 1000 app
USER app

# As the app user, prove the saved index loads unchanged (no rebuild) and the cached
# model answers a search offline; any download or permission problem fails the build.
COPY deploy/check_index.py /tmp/check_index.py
RUN python /tmp/check_index.py

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).read() == b'ok' else 1)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
