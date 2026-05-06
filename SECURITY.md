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
