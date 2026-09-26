# Service image: FastAPI in front of an Ollama server (the model runs outside this container).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

RUN useradd -r -u 10001 app
USER app
EXPOSE 8072
CMD ["uv", "run", "--no-sync", "uvicorn", "call_summary.service:create_app", "--factory", "--host", "0.0.0.0", "--port", "8072"]
