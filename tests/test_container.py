"""컨테이너 파일의 정적 검사(테스트는 Docker 없이 돈다)."""

# [테스트 묶음] Dockerfile과 compose.yaml을 글로 읽어 검사한다. 테스트는 Docker 없이 돌아야 하므로 이미지를
# 만들지 않고, 예전에 고친 두 문제가 다시 들어오지 않는지만 본다(커밋 966276e, 82ecfdb).

import json
import re
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
