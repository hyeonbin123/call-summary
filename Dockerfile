# 서비스 이미지: Ollama 서버 앞에 서는 FastAPI다. 모델은 이 컨테이너 밖(호스트의 Ollama)에서 돈다.
#
# 판 고정: 베이스 이미지와 uv 이미지는 태그 뒤에 digest(@sha256:...)를 붙여 고정한다. 태그는 나중에 다른 이미지로
# 옮겨질 수 있어, 태그만 쓰면 같은 Dockerfile이 빌드할 때마다 다른 파이썬 패치 판·데비안 패키지·uv 판을 받는다.
# digest가 있으면 태그는 읽는 사람을 위한 표시이고 받는 것은 digest가 가리키는 이미지다. 두 digest 모두 플랫폼별
# 이미지를 묶은 목록(OCI image index)의 것이라 amd64·arm64 어디서 빌드해도 같은 판을 받는다.
# - python:3.11-slim: 2026-10-29에 고정. Python 3.11.18, Debian 13 trixie (이미지 표시 3.11.18-slim-trixie)
# - uv 0.13.1: CI(setup-uv의 version)와 같은 판이고 pyproject.toml의 required-version 하한 이상이어야 한다
#   (tests/test_container.py가 셋을 함께 본다)
# 올릴 때는 `docker buildx imagetools inspect <이미지>:<태그>`의 맨 위 Digest로 바꾸고 위 판 설명도 함께 고친다.
FROM python:3.11-slim@sha256:673719e071ecf27434fd2635bbd8d00f49f4fdd81f243143438a42ced8bedf60

ENV PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"
COPY --from=ghcr.io/astral-sh/uv:0.13.1@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

RUN useradd -r -u 10001 app
USER app
EXPOSE 8072
# Start from the venv, not through `uv run`: the app user has no home directory for uv's cache.
CMD ["uvicorn", "call_summary.service:create_app", "--factory", "--host", "0.0.0.0", "--port", "8072"]
