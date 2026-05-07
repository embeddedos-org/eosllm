# Release pipeline

This document is the operator manual for the **engine** release
pipeline. (Editor and browser-extension publishes have their own
docs: `vscode-extension/README.md`, `browser-extension/README.md`.)

## TL;DR

```
git tag v0.2.0
git push origin v0.2.0
```

GitHub Actions then runs `.github/workflows/release.yml`, which:

1. resolves the version from the tag,
2. builds Linux x86_64, Linux aarch64, macOS universal (arm64+x86_64),
   and Windows x86_64 binaries in parallel jobs,
3. packages a source tarball,
4. computes `SHA256SUMS` over every artifact,
5. (optional) signs `SHA256SUMS` with minisign,
6. publishes a single GitHub Release with everything attached.

The release is created as `draft: false`. If the tag contains `-`
(e.g. `v0.2.0-rc1`) it's flagged as a `prerelease`.

## Required GitHub repo secrets

The pipeline runs without any secrets — but the binaries it ships will
be **unsigned** and the release notes will say so. To enable each
signing path, add the corresponding secret in
**Settings → Secrets and variables → Actions**.

| Secret name | Purpose | Required for |
|---|---|---|
| `MINISIGN_KEY` | Encrypted minisign secret key (full file contents, including the `untrusted comment:` lines) | Signing `SHA256SUMS` |
| `MINISIGN_PASSWORD` | Passphrase that decrypts `MINISIGN_KEY` | Signing `SHA256SUMS` |
| `APPLE_DEVELOPER_ID_CERT_P12_B64` | Base64-encoded Developer ID Application certificate (`base64 -i Cert.p12 \| pbcopy`) | macOS codesign + notarization |
| `APPLE_DEVELOPER_ID_PASSWORD` | Passphrase that decrypts the `.p12` | macOS codesign |
| `APPLE_TEAM_ID` | Apple Developer Team ID (10-char) | macOS codesign + notarization |
| `APPLE_NOTARY_KEY_ID` | App Store Connect API key ID (10-char) | macOS notarization |
| `APPLE_NOTARY_ISSUER_ID` | App Store Connect issuer ID (UUID) | macOS notarization |
| `APPLE_NOTARY_KEY_P8_B64` | Base64 of the `.p8` private key file | macOS notarization |
| `WINDOWS_CERT_PFX_B64` | Base64-encoded Authenticode `.pfx` | Windows signing |
| `WINDOWS_CERT_PASSWORD` | Passphrase for the `.pfx` | Windows signing |

Each secret is **optional**. The corresponding step is gated on the
secret's presence; missing secrets cause the workflow to ship that
platform's binary unsigned (and only that platform).

## Per-platform jobs

### `build-linux-x86_64`

Runs on `ubuntu-latest`. Standard host build. The Makefile auto-detects
AVX2 from `/proc/cpuinfo` and builds with `-mavx2 -mfma`. Runs
`make all-checks` (the canonical health gate, ~5 min) before staging
the tarball.

### `build-linux-aarch64`

Runs on `ubuntu-latest`. Cross-compile with `gcc-aarch64-linux-gnu`
plus `-static`. Runs the unit suite under `qemu-user-static` to catch
endianness or NEON intrinsic regressions.

### `build-macos-universal`

Runs on `macos-14` (the only macos GHA runner image with an
arm64-native build environment as of this writing).
`tools/release/build_macos_universal.sh` builds the engine twice
(`-target arm64-apple-macos11` and `-target x86_64-apple-macos11`)
then `lipo`-merges every binary + `libeosllm.a`. If the Apple secrets
are present, `tools/release/sign_and_notarize_macos.sh` codesigns each
binary, submits to `notarytool`, and waits for the ticket.

### `build-windows-x86_64`

Runs on `windows-latest` under `msys2/setup-msys2@v2` with the
`MINGW64` environment. Builds with the `EOSLLM_HAVE_WIN32` shim
enabled and POSIX disabled, statically links against `libgcc` so the
binary runs on a vanilla Windows install. If `WINDOWS_CERT_PFX_B64`
is set, `tools/release/sign_windows.ps1` Authenticode-signs every
`.exe` with SHA-256 + RFC 3161 timestamp.

### `build-source-tarball`

`git archive` of `HEAD` with prefix `eosllm-${version}/`. Same as the
old single-job workflow.

### `wasm-stub`

Currently `if: false` — the slot is reserved for Phase F (WASM
matmul kernel + Emscripten OS shim + IndexedDB model cache). When
implemented, flip the gate and the WASM artifact joins the release
matrix.

### `publish-release`

Aggregator job. Downloads every other job's artifacts, computes
`SHA256SUMS`, optionally minisigns it, and posts the GitHub Release.
The actual `softprops/action-gh-release@v2` step is gated on
`startsWith(github.ref, 'refs/tags/v')` so `workflow_dispatch` runs
build everything but skip publication.

## Verification

### Hash check

```
sha256sum --check SHA256SUMS
```

### minisign (optional)

```
# eosllm-release.pub fingerprint is published in SECURITY.md.
minisign -V -p eosllm-release.pub -m SHA256SUMS
```

### macOS codesign

```
codesign --verify --deep --strict --verbose=2 eosllm-cli
spctl --assess --type execute --verbose eosllm-cli
```

### Windows signtool

```
signtool verify /pa /v eosllm-cli.exe
```

## Unsigned binaries

If `SHA256SUMS` doesn't have a `SHA256SUMS.minisig` next to it, the
release is **not** verifiable beyond the hash. If you trust the
`HTTPS GitHub Release URL`, the SHA256 is sufficient; if you don't,
build from source via the source tarball (the build is reproducible).

### macOS Gatekeeper bypass (unsigned releases only)

```
xattr -dr com.apple.quarantine eosllm-cli
```

…or right-click → *Open* the first time. Both bypass Gatekeeper for
the specific binary; do not blanket-disable Gatekeeper.

### Windows SmartScreen (unsigned releases only)

SmartScreen will warn on first launch. Click **More info → Run anyway**.
Microsoft's reputation system clears unsigned binaries after enough
unique installs, but for a fresh release the warning is expected.

## Cutting a tag

```
# 1. Update CHANGELOG.md — move [Unreleased] entries under [vX.Y.Z]
#    with today's date.
# 2. Run the local health gate one last time.
make all-checks

# 3. Tag and push.
git tag v0.2.0 -m "v0.2.0 — multi-platform release"
git push origin v0.2.0
```

The workflow will start within seconds. Watch progress under the
**Actions** tab; if any platform job fails the publish-release job
won't trigger.

## Cutting a release-candidate

```
git tag v0.2.0-rc1
git push origin v0.2.0-rc1
```

The same workflow runs; the GitHub Release is flagged
`prerelease: true` (so it doesn't show up as the "latest" release).

## Rollback / unpublish

```
gh release delete v0.2.0 --yes
git push --delete origin v0.2.0
git tag --delete v0.2.0
```

The artifacts are deleted from the Release page; the build artifacts
(uploaded via `actions/upload-artifact`) are GC'd by GitHub after 90
days regardless.

## Future: shipping a separate update channel

Each subdirectory has its own tag-driven publisher workflow:

- Engine: `vX.Y.Z`            → `release.yml` (this doc)
- VS Code extension: `vscode-vX.Y.Z` → `release-vscode-extension.yml`
- Browser extension: `browser-vX.Y.Z` → `release-browser-extension.yml`

The three are independent so a hot-fix to one doesn't force a re-tag
of the others.
