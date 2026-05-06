"""eosllm-convert CLI entry point.

`from-gguf` shells out to the C-side `eosllm-convert` binary (built
by `make convert`). `from-hf` and `calibrate` remain Phase 2 stubs.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


def _find_convert_binary() -> str | None:
    """Locate the C-side binary. Searches:
       1. $EOSLLM_CONVERT_BIN env override
       2. tools/eosllm-convert/eosllm-convert relative to this file
       3. PATH
    """
    env = os.environ.get("EOSLLM_CONVERT_BIN")
    if env and os.path.isfile(env) and os.access(env, os.X_OK):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.normpath(os.path.join(here, "..", "..", ".."))
    candidate = os.path.join(repo_root, "tools", "eosllm-convert", "eosllm-convert")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return shutil.which("eosllm-convert")


def cmd_from_hf(args: argparse.Namespace) -> int:
    print(
        f"[eosllm-convert] from-hf is a Phase 2 feature.\n"
        f"  Would convert: {args.source}\n"
        f"  Quant scheme : {args.quant}\n"
        f"  Output       : {args.out}",
        file=sys.stderr,
    )
    return 1


def cmd_from_gguf(args: argparse.Namespace) -> int:
    binary = _find_convert_binary()
    if binary is None:
        print(
            "[eosllm-convert] cannot find the C-side eosllm-convert binary.\n"
            "  Build it first:  (cd <repo> && make convert)\n"
            "  Or set EOSLLM_CONVERT_BIN to its path.",
            file=sys.stderr,
        )
        return 1
    cmd = [binary, "from-gguf", args.source, "-o", args.out]
    print(f"[eosllm-convert] running: {' '.join(cmd)}", file=sys.stderr)
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except OSError as exc:
        print(f"[eosllm-convert] failed to launch {binary!r}: {exc}",
              file=sys.stderr)
        return 1


def cmd_calibrate(args: argparse.Namespace) -> int:
    print(
        f"[eosllm-convert] calibrate is a Phase 2 feature.\n"
        f"  Quant scheme : {args.quant}\n"
        f"  Dataset      : {args.dataset}",
        file=sys.stderr,
    )
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eosllm-convert")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_hf = sub.add_parser("from-hf", help="Convert a HuggingFace checkpoint.")
    p_hf.add_argument("source", help="HF repo id or local checkpoint path.")
    p_hf.add_argument("--quant", default="q4_k",
                      choices=["q8_0", "q4_k", "q2_k", "q1_58", "mixed", "calibrated"])
    p_hf.add_argument("-o", "--out", required=True)
    p_hf.set_defaults(func=cmd_from_hf)

    p_gg = sub.add_parser("from-gguf", help="Lossless pass-through of a .gguf file.")
    p_gg.add_argument("source")
    p_gg.add_argument("-o", "--out", required=True)
    p_gg.set_defaults(func=cmd_from_gguf)

    p_cal = sub.add_parser("calibrate", help="Run AWQ-style calibration.")
    p_cal.add_argument("--quant", required=True)
    p_cal.add_argument("--dataset", required=True)
    p_cal.set_defaults(func=cmd_calibrate)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
