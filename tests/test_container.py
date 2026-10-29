"""컨테이너 파일의 정적 검사(테스트는 Docker 없이 돈다)."""

# [테스트 묶음] Dockerfile과 compose.yaml(그리고 uv 판을 적는 CI 워크플로·pyproject.toml)을 글로 읽어
# 검사한다. 테스트는 Docker 없이 돌아야 하므로 이미지를 만들지 않고, 예전에 고친 문제(커밋 966276e, 82ecfdb)와
# 판 고정이 다시 풀리지 않는지만 본다. CI의 테스트 환경에는 YAML 파서가 없어 YAML은 정규식으로 읽는다.

import json
import re
import tomllib
from pathlib import Path

# 저장소 루트: tests/의 한 단계 위.
ROOT = Path(__file__).resolve().parents[1]


# 컨테이너는 홈 디렉터리가 없는 시스템 사용자(app)로 돈다. 이때 `uv run`은 캐시 폴더를 만들지 못해 컨테이너가
# 바로 죽었다(커밋 966276e). 그래서 CMD의 첫 낱말이 uvicorn이고, 가상환경의 bin이 PATH 앞에 있는지
# 확인한다(가상환경의 uvicorn이 실행되도록).
def test_image_starts_uvicorn_from_the_venv_not_through_uv():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    cmd = re.search(r"^CMD (\[.*\])$", dockerfile, re.M)
    assert cmd and json.loads(cmd.group(1))[0] == "uvicorn"
    assert re.search(r'^ENV .*PATH="/app/\.venv/bin:\$PATH"', dockerfile, re.M)


# API에는 인증이 없고 요청마다 공유 GPU를 쓰므로, compose가 포트를 127.0.0.1에만 열어야 한다(커밋 82ecfdb).
# ports 블록의 모든 항목이 "127.0.0.1:"로 시작하는지 본다. 항목이 하나도 없어도 실패한다.
def test_compose_publishes_the_api_on_loopback_only():
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    block = re.search(r"^ *ports:\n((?: *(?:#.*|- .*)\n)+)", compose, re.M)
    ports = re.findall(r'- "([^"]+)"', block.group(1)) if block else []
    assert ports and all(p.startswith("127.0.0.1:") for p in ports)


# 베이스 이미지(FROM)와 uv를 꺼내 오는 이미지(COPY --from)는 태그 뒤에 digest를 붙여 고정한다. 태그는 나중에
# 다른 이미지로 옮겨질 수 있어, 태그만 쓰면 같은 Dockerfile이 빌드할 때마다 다른 파이썬 패치 판·데비안
# 패키지·uv 판을 받는다. `이름:태그@sha256:<64자리>` 꼴만 허용한다(태그는 사람이 읽는 표시이고 받는 것은
# digest가 가리키는 이미지). 이미지 참조가 하나도 없어도 실패한다.
def test_base_and_uv_images_are_pinned_by_digest():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    froms = re.findall(r"^FROM (\S+)", dockerfile, re.M)
    copies = re.findall(r"^COPY --from=(\S+)", dockerfile, re.M)
    assert froms and copies
    for ref in froms + copies:
        assert re.fullmatch(r"\S+:[\w.-]+@sha256:[0-9a-f]{64}", ref), ref


# "0.13.1" 같은 판 글을 비교할 수 있는 정수 튜플로 바꾼다.
def _version(text):
    return tuple(int(part) for part in text.split("."))


# uv 판을 정하는 세 곳이 맞물려야 한다: 이미지에 넣는 uv(Dockerfile), CI가 설치하는 uv(작업마다 setup-uv의
# `version`), pyproject.toml `[tool.uv] required-version`의 하한(이보다 낮은 uv는 이 저장소에서 명령을
# 거부한다). 전에는 이미지 0.12 계열, CI 최신, 개발 PC 0.13.1로 셋이 달라 같은 uv.lock을 서로 다른 uv가
# 읽었다. 하한은 `>=`만 쓴다: `~=`처럼 위를 막으면 개발 PC의 uv를 올리는 순간 이 저장소의 모든 uv 명령이
# 멈춘다. 그래서 이미지와 CI의 판이 모두 같고 그 판이 하한 이상인지 본다. setup-uv는 `version`이 없으면
# required-version 범위에서 가장 높은 판(곧 최신)을 받으므로 setup-uv 단계마다 `version`이 있어야
# 한다(setup-uv README "Install a required-version or latest").
def test_uv_version_is_the_same_in_the_image_and_ci_and_meets_the_project_floor():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    image = re.search(r"^COPY --from=ghcr\.io/astral-sh/uv:(\d+\.\d+\.\d+)@", dockerfile, re.M)
    assert image
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    # setup-uv 단계 하나 = `- uses: astral-sh/setup-uv@...` 줄과, 그보다 두 칸 더 들여 쓴 줄·빈 줄이 이어지는
    # 데까지 (같은 들여쓰기의 다음 `- ` 단계나 주석 줄에서 끝난다).
    steps = re.findall(r"^( *)- uses: astral-sh/setup-uv@.*\n((?:\1  .*\n|[ \t]*\n)*)", ci, re.M)
    assert steps
    ci_versions = [re.search(r'^ +version: "(\d+\.\d+\.\d+)"$', body, re.M) for _, body in steps]
    assert all(ci_versions)
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    floor = re.fullmatch(r">=(\d+\.\d+\.\d+)", pyproject["tool"]["uv"]["required-version"])
    assert floor
    assert {m.group(1) for m in ci_versions} == {image.group(1)}
    assert _version(image.group(1)) >= _version(floor.group(1))
