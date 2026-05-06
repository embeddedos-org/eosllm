#!/bin/bash
# tools/check_includes.sh — architectural firewall: no source file in
# src/ may include from tests/, tools/, or build/. (Headers in
# include/ and other files in src/ are fine.)
#
# Exits 0 if the firewall holds, 1 otherwise.
set -uo pipefail
cd "$(dirname "$0")/.."

bad=$(grep -rEn '#include[[:space:]]*[<"](\.\./)*tests/' src/ 2>/dev/null || true)
bad="${bad}${bad:+\n}$(grep -rEn '#include[[:space:]]*[<"](\.\./)*tools/' src/ 2>/dev/null || true)"
bad="${bad}${bad:+\n}$(grep -rEn '#include[[:space:]]*[<"](\.\./)*build/' src/ 2>/dev/null || true)"

n_files=$(find src -name '*.c' -o -name '*.h' | wc -l)

if [ -n "$bad" ]; then
  echo "FAIL: src/ includes leak into tests/ / tools/ / build/:"
  echo -e "$bad" | sed 's/^/  /'
  exit 1
fi

echo "include-firewall: OK ($n_files files in src/, none reach into tests/tools/build)"
