"""Static checks of the container files (the tests run without Docker)."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_image_starts_uvicorn_from_the_venv_not_through_uv():
    # The image runs as a system user with no home directory; `uv run` fails there creating its cache.
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    cmd = re.search(r"^CMD (\[.*\])$", dockerfile, re.M)
    assert cmd and json.loads(cmd.group(1))[0] == "uvicorn"
    assert re.search(r'^ENV .*PATH="/app/\.venv/bin:\$PATH"', dockerfile, re.M)
