"""eosllm-quant-lab CLI scaffold (Phase 2 deliverable)."""
from __future__ import annotations
import argparse, sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="eosllm-quant-lab")
    p.add_argument("--quant", required=True)
    p.add_argument("--dataset", default="wikitext-2")
    p.add_argument("--report", default="-")
    args = p.parse_args(argv)
    print(f"[eosllm-quant-lab] Phase 2 deliverable. Args: {vars(args)}",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
