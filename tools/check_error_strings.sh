#!/bin/bash
# tools/check_error_strings.sh — verify every EOSI_LOG_ERROR("...")
# string in src/ is unique. Useful so eos_last_error() always tells
# you exactly which site fired.
#
# Exits 0 if all strings are unique, 1 if any duplicates exist.
set -uo pipefail
cd "$(dirname "$0")/.."

# Extract every literal first-arg string from EOSI_LOG_ERROR(...). The
# regex is single-line so multi-line concatenated string literals (the
# "..." "..." pattern inside one EOSI_LOG_ERROR call) get captured as
# their first segment, which is fine for uniqueness purposes.
all=$(grep -rhE 'EOSI_LOG_ERROR\("[^"]+"' src/ \
      | sed -E 's/.*EOSI_LOG_ERROR\("([^"]+)".*/\1/')

n_total=$(echo "$all" | wc -l)
dups=$(echo "$all" | sort | uniq -d)
n_dups=$(if [ -z "$dups" ]; then echo 0; else echo "$dups" | wc -l; fi)

echo "EOSI_LOG_ERROR sites: $n_total"
echo "duplicate strings:    $n_dups"

if [ "$n_dups" -ne 0 ]; then
  echo ""
  echo "FAIL: the following strings appear at multiple sites:"
  echo "$dups" | sed 's/^/  /'
  exit 1
fi

echo "OK: every error site has a unique message."
