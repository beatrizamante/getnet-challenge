# ---- builder ----
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
ENV UV_HTTP_TIMEOUT=120
RUN uv sync --frozen --no-dev --no-install-project

# ---- runtime ----
FROM python:3.13-slim-bookworm

# Non-root user — principle of least privilege
RUN groupadd --system app && useradd --system --gid app --create-home app

WORKDIR /app

COPY --from=builder /app/.venv .venv
COPY src/ src/
COPY scripts/ scripts/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME="/app/hf_cache"

RUN chown -R app:app /app
USER app

EXPOSE 8080

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
