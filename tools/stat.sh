#!/bin/bash
# tools/stat.sh — print one-shot headline numbers about the engine.
# Used by `make stat`. Designed for human + grep-friendly output.
set -uo pipefail
cd "$(dirname "$0")/.."

n_src=$(find src -name '*.c' -o -name '*.h' | wc -l)
n_include=$(find include -name '*.h' | wc -l)
n_log_err=$(grep -rhE 'EOSI_LOG_ERROR\("[^"]+"' src/ \
            | sed -E 's/.*EOSI_LOG_ERROR\("([^"]+)".*/\1/' | wc -l)
n_smoke=$(grep -cE '^\s*echo "--- ' Makefile)
n_ci_jobs=$(grep -cE '^  [a-z][a-z0-9-]*:$' .github/workflows/ci.yml || echo 0)
n_tools=$(find tools -name '*.c' | wc -l)
n_fuzz=$(find tests/fuzz -name 'fuzz_*.c' | wc -l)

# Test count: best to actually run.
test_out=$(make -s test 2>/dev/null | tail -1 || echo "?/? checks passed")
n_tests=$(echo "$test_out" | grep -oE '[0-9]+/' | head -1 | tr -d '/')

# Lines of code in src/ + include/ + tools/ (excluding generated).
loc_src=$(find src include -name '*.c' -o -name '*.h' | xargs cat 2>/dev/null | wc -l)
loc_tools=$(find tools -name '*.c' -o -name '*.py' -o -name '*.sh' | xargs cat 2>/dev/null | wc -l)
loc_tests=$(find tests -name '*.c' | xargs cat 2>/dev/null | wc -l)

cat <<EOF
eosllm engine — headline numbers

  source files (src/):       $n_src
  public headers (include/): $n_include
  tool source files:         $n_tools
  fuzz harnesses:            $n_fuzz

  unit assertions:           ${n_tests:-?}
  EOSI_LOG_ERROR sites:      $n_log_err
  smoke-all variants:        $n_smoke
  CI PR-gating jobs:         $n_ci_jobs

  LOC src + include:         $loc_src
  LOC tools:                 $loc_tools
  LOC tests:                 $loc_tests
EOF
