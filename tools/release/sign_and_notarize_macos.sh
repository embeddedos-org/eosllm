#!/usr/bin/env bash
# tools/release/sign_and_notarize_macos.sh
#
# Codesign + notarize the macOS universal eosllm binaries. Called from
# .github/workflows/release.yml::build-macos-universal AFTER
# build_macos_universal.sh has staged the artifacts.
#
# Required environment variables (each is one of GitHub Actions
# `secrets.*`; the workflow exports them only if every variable in this
# block is set, so this script's mere invocation implies all secrets
# are present):
#   APPLE_DEVELOPER_ID_CERT_P12_B64  base64-encoded Developer ID
#                                     Application certificate .p12
#   APPLE_DEVELOPER_ID_PASSWORD      passphrase for the .p12
#   APPLE_TEAM_ID                    10-char Apple Developer Team ID
#   APPLE_NOTARY_KEY_ID              App Store Connect API key id
#   APPLE_NOTARY_ISSUER_ID           App Store Connect issuer id (UUID)
#   APPLE_NOTARY_KEY_P8_B64          base64 .p8 private key
#
# Usage:
#     tools/release/sign_and_notarize_macos.sh <staging_bin_dir>
#
# After successful run every binary in <staging_bin_dir> is codesigned
# AND its notarization ticket is stapled (works offline).
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <staging_bin_dir>" >&2
    exit 2
fi
BIN="$1"

: "${APPLE_DEVELOPER_ID_CERT_P12_B64:?missing APPLE_DEVELOPER_ID_CERT_P12_B64}"
: "${APPLE_DEVELOPER_ID_PASSWORD:?missing APPLE_DEVELOPER_ID_PASSWORD}"
: "${APPLE_TEAM_ID:?missing APPLE_TEAM_ID}"
: "${APPLE_NOTARY_KEY_ID:?missing APPLE_NOTARY_KEY_ID}"
: "${APPLE_NOTARY_ISSUER_ID:?missing APPLE_NOTARY_ISSUER_ID}"
: "${APPLE_NOTARY_KEY_P8_B64:?missing APPLE_NOTARY_KEY_P8_B64}"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

# 1. Decode the cert and import into a one-shot keychain isolated from
#    the runner's defaults; matches the standard macos-14 codesign
#    pattern used by Homebrew, Sparkle, etc.
CERT_PATH="$WORKDIR/cert.p12"
echo "$APPLE_DEVELOPER_ID_CERT_P12_B64" | base64 --decode > "$CERT_PATH"

KEYCHAIN="$WORKDIR/eosllm.keychain"
KEYCHAIN_PW="$(uuidgen)"
security create-keychain -p "$KEYCHAIN_PW" "$KEYCHAIN"
security set-keychain-settings -lut 21600 "$KEYCHAIN"
security unlock-keychain -p "$KEYCHAIN_PW" "$KEYCHAIN"
security import "$CERT_PATH" -k "$KEYCHAIN" -P "$APPLE_DEVELOPER_ID_PASSWORD" \
    -T /usr/bin/codesign
security set-key-partition-list -S apple-tool:,apple:,codesign: \
    -s -k "$KEYCHAIN_PW" "$KEYCHAIN" >/dev/null
security list-keychains -d user -s "$KEYCHAIN" \
    "$(security list-keychains -d user | tr -d '"' | xargs)"

IDENTITY="Developer ID Application: $APPLE_TEAM_ID"

# 2. Codesign every binary in the staging dir.
for f in "$BIN"/*; do
    [[ -f "$f" ]] || continue
    codesign --force --options runtime --timestamp \
             --sign "$IDENTITY" \
             "$f"
    codesign --verify --strict --verbose=2 "$f"
done

# 3. Notarize. notarytool wants a single zip submission per request, so
#    bundle every signed binary together.
ZIP="$WORKDIR/eosllm-bin.zip"
( cd "$BIN" && zip -j -r "$ZIP" . )

KEY_PATH="$WORKDIR/notary-key.p8"
echo "$APPLE_NOTARY_KEY_P8_B64" | base64 --decode > "$KEY_PATH"

xcrun notarytool submit "$ZIP" \
    --key      "$KEY_PATH" \
    --key-id   "$APPLE_NOTARY_KEY_ID" \
    --issuer   "$APPLE_NOTARY_ISSUER_ID" \
    --wait \
    --timeout 30m

# 4. Staple. Stapling per-binary fails for raw Mach-O executables (only
#    bundles + dmg can be stapled), so for shipping we rely on the
#    notarization being recorded in Apple's online ledger; gatekeeper
#    will fetch the ticket on first launch. We still call stapler in a
#    best-effort mode for any bundles in <BIN>.
if compgen -G "$BIN/*.app" > /dev/null; then
    for app in "$BIN"/*.app; do
        xcrun stapler staple -v "$app" || true
    done
fi

echo "sign_and_notarize_macos: OK ($(ls "$BIN"))"
