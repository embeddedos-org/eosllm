#!/usr/bin/env bash
# tools/fetch_test_models.sh
#
# Idempotently download model files used by the golden / oracle test
# suites into tests/data/. Verifies SHA-256 after download. Re-running
# is a no-op when files are already present and intact.
#
# Usage:
#   ./tools/fetch_test_models.sh             # all known models
#   ./tools/fetch_test_models.sh tinyllama   # one model by short name
#   ./tools/fetch_test_models.sh --list      # show known models
#
# Requires: bash, curl, sha256sum.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${REPO_ROOT}/tests/data"
mkdir -p "${DATA_DIR}"

# ---------------------------------------------------------------------
# Model registry: NAME|URL|SHA256|DEST_FILENAME
# Lines starting with # are ignored.
# Set SHA256 to "-" to skip checksum verification (NOT recommended;
# included only for placeholders the user must fill in).
# ---------------------------------------------------------------------
read -r -d '' MODELS <<'EOF' || true
tinyllama|https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf|-|tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
EOF

list_models() {
    echo "Known models:"
    while IFS='|' read -r name url sha dest; do
        [[ -z "$name" || "$name" == \#* ]] && continue
        echo "  ${name}  -> tests/data/${dest}"
    done <<< "${MODELS}"
}

fetch_one() {
    local name="$1"
    local found=0
    while IFS='|' read -r m_name url sha dest; do
        [[ -z "$m_name" || "$m_name" == \#* ]] && continue
        if [[ "$m_name" == "$name" ]]; then
            found=1
            local path="${DATA_DIR}/${dest}"
            if [[ -f "$path" && "$sha" != "-" ]]; then
                local have
                have="$(sha256sum "$path" | awk '{print $1}')"
                if [[ "$have" == "$sha" ]]; then
                    echo "[ok] ${dest} already present, checksum verified"
                    return 0
                fi
                echo "[warn] ${dest} present but checksum mismatch; re-downloading"
                rm -f "$path"
            elif [[ -f "$path" && "$sha" == "-" ]]; then
                echo "[ok] ${dest} already present (no checksum to verify)"
                return 0
            fi
            echo "[fetch] ${dest}  <- ${url}"
            curl -L --fail --progress-bar -o "${path}.part" "$url"
            mv "${path}.part" "$path"
            if [[ "$sha" != "-" ]]; then
                local have
                have="$(sha256sum "$path" | awk '{print $1}')"
                if [[ "$have" != "$sha" ]]; then
                    echo "[fail] checksum mismatch for ${dest}" >&2
                    echo "  expected: ${sha}" >&2
                    echo "  got     : ${have}" >&2
                    rm -f "$path"
                    return 1
                fi
                echo "[ok] checksum verified"
            else
                echo "[ok] downloaded (no checksum recorded; please update fetch_test_models.sh)"
                echo "     sha256: $(sha256sum "$path" | awk '{print $1}')"
            fi
            return 0
        fi
    done <<< "${MODELS}"
    if (( found == 0 )); then
        echo "[fail] unknown model: $name" >&2
        list_models >&2
        return 1
    fi
}

main() {
    if [[ $# -eq 0 ]]; then
        # Fetch every model in the registry.
        while IFS='|' read -r name url sha dest; do
            [[ -z "$name" || "$name" == \#* ]] && continue
            fetch_one "$name"
        done <<< "${MODELS}"
        return 0
    fi
    case "$1" in
        --list|-l) list_models; return 0 ;;
        --help|-h) sed -n '2,15p' "${BASH_SOURCE[0]}"; return 0 ;;
        *)         fetch_one "$1" ;;
    esac
}

main "$@"
