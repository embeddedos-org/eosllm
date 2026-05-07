#!/usr/bin/env bash
# tools/validate_release_pipeline.sh
# Validates JSON + YAML files added by the production release pipeline.
# Exits non-zero on the first parse failure.
set -e
cd "$(dirname "$0")/.."

echo "--- YAML workflows ---"
for f in .github/workflows/*.yml; do
    python3 -c 'import yaml,sys; yaml.safe_load(open(sys.argv[1])); print("OK", sys.argv[1])' "$f"
done

echo "--- JSON manifests / configs ---"
for f in vscode-extension/package.json \
         vscode-extension/tsconfig.json \
         browser-extension/package.json \
         browser-extension/manifest.json \
         browser-extension/tsconfig.json; do
    python3 -c 'import json,sys; json.load(open(sys.argv[1])); print("OK", sys.argv[1])' "$f"
done

echo "validate_release_pipeline: OK"
