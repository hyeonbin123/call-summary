"""Write the spec JSONL of one split. Specs are deterministic, so this can be rerun at any time."""

from __future__ import annotations

import argparse

from .dataset import write_jsonl
from .specs import SPLIT_DOMAINS, make_specs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, choices=sorted(SPLIT_DOMAINS))
    ap.add_argument("--per-domain", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    specs = make_specs(args.split, SPLIT_DOMAINS[args.split], args.per_domain)
    n = write_jsonl(args.out, (s.to_dict() for s in specs))
    print(f"{n} specs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
