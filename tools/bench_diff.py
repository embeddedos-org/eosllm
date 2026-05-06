#!/usr/bin/env python3
"""tools/bench_diff.py — compare two `eosllm-bench` JSON outputs.

Usage:
    tools/bench_diff.py <baseline.json> <candidate.json> [--threshold-pct=10]

Joins records by (backend, op, m, n, k) and reports the percent change
in giga_ops_per_s of candidate vs baseline. Exits non-zero if any
common record regresses beyond the threshold (default 10%).

Records present in only one side are reported but do not fail the
diff (so adding a new backend is not a regression).

The output is a table on stdout; the exit code is the actionable
signal for CI.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def load(path: str) -> dict[tuple, dict]:
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    out: dict[tuple, dict] = {}
    for r in doc.get("records", []):
        key = (r["backend"], r["op"], r["m"], r["n"], r["k"])
        out[key] = r
    return out


def fmt_row(key: tuple, base: float | None, cand: float | None,
            delta_pct: float | None, status: str) -> str:
    backend, op, m, n, k = key
    bs = "      —" if base is None else f"{base:7.2f}"
    cs = "      —" if cand is None else f"{cand:7.2f}"
    ds = "      —" if delta_pct is None else f"{delta_pct:+6.1f}%"
    return (
        f"  {backend:<10} {op:<14} m={m:<3} n={n:<4} k={k:<5}  "
        f"baseline={bs}  candidate={cs}  delta={ds}  {status}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--threshold-pct", type=float, default=10.0,
                   help="Per-record regression threshold (default 10%%).")
    args = p.parse_args(argv)

    base = load(args.baseline)
    cand = load(args.candidate)

    all_keys = sorted(set(base) | set(cand))
    n_records = 0
    n_regressed = 0
    n_only_base = 0
    n_only_cand = 0

    print(f"\nbench-diff: {args.baseline}  ->  {args.candidate}")
    print(f"            threshold = {args.threshold_pct:.1f}% regression\n")

    for key in all_keys:
        b = base.get(key)
        c = cand.get(key)
        if b is None:
            print(fmt_row(key, None, c["giga_ops_per_s"], None, "[NEW]"))
            n_only_cand += 1
            continue
        if c is None:
            print(fmt_row(key, b["giga_ops_per_s"], None, None, "[REMOVED]"))
            n_only_base += 1
            continue
        n_records += 1
        bv = float(b["giga_ops_per_s"])
        cv = float(c["giga_ops_per_s"])
        if bv == 0:
            delta_pct = 0.0
        else:
            delta_pct = (cv - bv) * 100.0 / bv
        if delta_pct < -args.threshold_pct:
            status = "REGRESSED"
            n_regressed += 1
        elif delta_pct > args.threshold_pct:
            status = "improved"
        else:
            status = "ok"
        print(fmt_row(key, bv, cv, delta_pct, status))

    print()
    print(f"summary: {n_records} compared, "
          f"{n_regressed} regressed (>{args.threshold_pct:.1f}%), "
          f"{n_only_cand} new, {n_only_base} removed")

    if n_regressed > 0:
        print(f"\nFAIL: {n_regressed} record(s) regressed beyond threshold.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
