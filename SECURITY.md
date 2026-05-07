# Security policy

Thanks for taking the time to disclose security issues responsibly.

## Supported versions

eosllm is pre-1.0; only the latest tagged release is supported. The
`main` branch is the development tip and may break ABI between commits
(see `docs/abi.md`).

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅ (current) |
| < 0.1   | ❌         |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security reports.**

Use one of these channels:

1. **GitHub private vulnerability reporting** (preferred):
   open the [Security advisories tab](https://github.com/embeddedos-org/eosllm/security/advisories)
   and click *Report a vulnerability*.
2. Email: `srpatcha@users.noreply.github.com` with subject prefix
   `[eosllm-security]`.

When reporting, please include:

- A clear description of the issue and its security impact.
- Reproducer (input bytes, harness command, expected vs observed).
- Affected commit SHA / version tag.
- Whether you'd like credit in the advisory.

## Response SLA

| Stage              | Target                        |
|--------------------|-------------------------------|
| Initial ack        | within 5 business days        |
| Triage + scope     | within 10 business days       |
| Fix or mitigation  | depends on severity (see below) |
| Public disclosure  | coordinated with reporter; default 90 days from triage |

Severity rubric (informal):

- **Critical** (RCE, sandbox escape, kernel-mode crash): mitigation
  within 14 days; backport to the latest tag.
- **High** (memory corruption, OOB write reachable from a `.gguf` /
  `.eosm` file): mitigation within 30 days.
- **Medium** (DoS via crafted input that doesn't corrupt state):
  fixed in the next minor release.
- **Low** (informational, hardening): tracked as a normal issue.

## Scope

In scope:

- The `eosllm` C library (`include/eosllm/`, `src/`) — including the
  GGUF reader, the `.eosm` reader/writer, the BPE tokenizer, the
  scheduler, the kernel dispatch, and the OS shim layer.
- The `eosllm-cli`, `eosllm-bench`, and `eosllm-convert` tools.
- The libFuzzer harnesses (`tests/fuzz/`).

Out of scope:

- Issues in third-party model files (the engine treats every byte of
  a `.gguf` / `.eosm` as untrusted; report file-corruption issues that
  cause **the engine** to misbehave, not the file's existence).
- Issues in build tooling (Make, GCC, Clang) themselves.
- Side-channel attacks against the host CPU or other tenants.

## Mitigations already in place

The engine treats every byte of every input file as untrusted.
Specific mitigations are tracked in `docs/threat_model.md` along with
the source file and test that enforces each. Highlights:

- 30 s/PR libFuzzer + ASan + UBSan + LeakSan against both `.eosm` and
  GGUF readers, with hand-built seed corpora.
- Audit-hardened GGUF v3 reader: overflow-safe tensor offset checks,
  exact pass-1 string-pool sizing, rejection of 0-rank tensors,
  non-power-of-two alignment, Q4_K-not-mod-256 element counts.
- `.eosm` SHA-256 trailer verified BEFORE any section parsing.
- Capability bits gate unsupported features at probe time (no
  silent skips).
- Every `EOSI_LOG_ERROR` site has a unique message
  (`make check-errors`) so `eos_last_error()` pinpoints the
  rejecting check by source line.

## Acknowledgments

Reporters who follow this policy will be credited in the resulting
GitHub Security Advisory unless they request anonymity.

## Release signing keys

Starting with **v0.2.0**, every release artifact is checksummed in
`SHA256SUMS` and (when the maintainer's signing keys are present)
also signed for offline verification:

- **`SHA256SUMS.minisig`** — detached
  [minisign](https://jedisct1.github.io/minisign/) signature over
  `SHA256SUMS`. Verifying with minisign requires no PGP keyring and
  works on every platform with a single small static binary.
  Public key fingerprint: **`(populated when v0.2.0 is cut)`**.

- **macOS binaries** are codesigned with
  *Developer ID Application: embeddedos-org* (Apple Team ID
  `(populated when v0.2.0 is cut)`) and notarized via Apple's
  notary service. Verify with:

  ```
  codesign --verify --deep --strict --verbose=2 eosllm-cli
  spctl --assess --type execute --verbose eosllm-cli
  ```

- **Windows binaries** are Authenticode-signed with a SHA-256 cert
  issued to *embeddedos-org* (cert thumbprint
  `(populated when v0.2.0 is cut)`) and timestamped via DigiCert's
  RFC 3161 server. Verify with:

  ```
  signtool verify /pa /v eosllm-cli.exe
  ```

The signing-key bootstrap procedure, secret-management policy, and
operator manual live in
[`docs/release.md`](docs/release.md).

### Key-rotation policy

- The minisign key has **no expiry**. We rotate when:
  - the maintainer's secure key storage is replaced, or
  - a private key is suspected of compromise.
- Rotation procedure: publish a one-time `KEYS-ROTATION` advisory
  containing both the old and new public keys + a minisign signature
  of the new key by the old key. Update `SECURITY.md` with the new
  fingerprint. The next release uses the new key only.
- The Apple Developer ID and Windows Authenticode certs have vendor
  expiries (Apple: 5 years; DigiCert: 1–3 years). Renewal happens
  at least 60 days before expiry; the new cert thumbprint is
  documented here in the same release that first uses it.

### Reporting a key compromise

If you suspect any signing key has been compromised, **report it
through the same channels as a vulnerability** (see *Reporting a
vulnerability* above). Do **not** open a public issue; the rotation
procedure must complete before the public advisory.
