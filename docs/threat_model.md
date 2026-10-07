# eosllm Threat Model (v0.1)

> **Status.** This is the v0.1 threat model. It scopes only what the
> shipped code does today: in-process C library, host-supplied OS
> shim, no network I/O, no privileged operations. Items deferred to
> v0.2+ are listed in [§9](#9-out-of-scope-for-v01).

## 1. Purpose

eosllm is an embedded LLM inference engine. It runs **on the host's
behalf**, in the host's process, with the host's privileges. This
document is the contract between the library and a security-conscious
host: what eosllm defends against, what it does not, and where the
host has to do its share.

It is also a checklist for reviewers: every mitigation listed in
[§6](#6-mitigations) maps to a specific source file and a specific
test that proves the mitigation is actually compiled in and exercised.

## 2. System scope

| In scope                                                                  | Out of scope                                                       |
|---------------------------------------------------------------------------|---------------------------------------------------------------------|
| The shipped C99 library and its public ABI under `include/eosllm/`.       | The host process, its memory, its other libraries.                 |
| The GGUF v3 reader (`src/format/gguf/`).                                  | GGUF version drift outside v3 (rejected at probe).                 |
| The BPE tokenizer (`src/tokenizer/bpe.c`).                                | SentencePiece (compiled out by default; future work).              |
| Scalar reference kernels (`src/kernels/scalar/`).                         | Vendor NPU SDKs (`src/kernels/npu/` is a v0.2 stub).               |
| The convert tool's lossless GGUF→`.eosm` path (when shipped).             | The Python convert tool's HF download path (host's environment).   |

## 3. Assets

In rough order of how bad it is to lose them:

1. **Host process integrity.** No malformed model file, no malformed
   prompt, no register-after-open race may corrupt host memory or
   redirect control flow.
2. **Host availability.** A malicious model file or pathological
   prompt must not cause unbounded allocation, infinite loops, or
   deadlocks within eosllm code paths.
3. **Output correctness.** A model file must either produce the same
   tokens it would on the reference implementation (`llama.cpp` for
   GGUF) or the loader must refuse it. Silent mis-tokenization is
   treated as a security bug, not a correctness one.
4. **Determinism.** Same build flags + same model + same seed + same
   prompt ⇒ bit-identical output. A regression here is reported and
   fixed before the next tag.

## 4. Adversary model

| Adversary                          | Capability                                                                    | Treated as in-scope?      |
|------------------------------------|-------------------------------------------------------------------------------|---------------------------|
| Untrusted model-file author        | Crafts a `.gguf` file the host loads via `eos_model_open`.                    | **Yes** — primary target. |
| Untrusted prompt source            | Submits arbitrary UTF-8 bytes via `eos_session_feed`.                         | **Yes**.                  |
| Untrusted module author            | Registers a hostile vt before the host opens its first session.               | Out — host's trust call.  |
| Co-resident process on shared host | Reads timing / cache side-channels from inference workload.                   | Out (see [§7](#7-residual-risks)). |
| Network attacker                   | The library performs **no** network I/O. N/A.                                 | N/A.                      |
| Physical attacker                  | Cold-boot, fault-injection, etc. Out of scope at this layer.                  | Out.                      |

## 5. Attack surface

### 5.1 Model file (highest-risk)

Two formats reach attacker-controlled bytes today:

1. **GGUF v3** — `eos_model_open(path, &model)` opens the file via the
   host's OS shim and parses every length / offset byte-by-byte in
   `gguf_open` (`src/format/gguf/gguf.c`).
2. **`.eosm` v1** — same entry point dispatches to `eosm_open`
   (`src/format/eosm/eosm.c`) when the magic matches. The eosm reader
   verifies the mandatory SHA-256 trailer **before** parsing any
   sections and rejects any capability bit not present in `eos_caps()`.

Anything either parser dereferences without a check is a direct
memory-corruption primitive.

### 5.2 Prompt (medium-risk)

`eos_session_feed(session, EOS_MODALITY_TEXT, data, len)` passes
attacker bytes to `bpe_encode` (`src/tokenizer/bpe.c`). The BPE
encoder allocates per-session scratch sized to the input length and
walks the merge table; bugs here can OOM the process or pin a CPU
indefinitely if a merge introduces a cycle.

### 5.3 Resource exhaustion

The host controls `eos_session_params_t.max_context` and
`arena_bytes`. A malicious caller can ask for a huge arena. Mitigation
is "the host's `eos_os_alloc` may refuse"; eosllm doesn't itself cap
allocation.

### 5.4 Module registry

Pre-`eos_session_open`, the host registers modules. Post-open the
registry is frozen. A bug that lets a module re-register after open
would corrupt in-flight session dispatch tables. We test the freeze.

## 6. Mitigations

Each row maps a mitigation to (a) the file that implements it and
(b) the test that proves it's there.

| Threat                                         | Mitigation                                                                                              | Code                                                | Test                                                                              |
|------------------------------------------------|---------------------------------------------------------------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------------------------------------|
| GGUF integer overflow on tensor offset         | Overflow-safe `t->offset` + `t->hdr.data_bytes` bounds check.                                           | `src/format/gguf/gguf.c` (resolve loop)             | `make sanitize` (UBSan would catch any wrap).                                     |
| GGUF over-allocation via inflated string length | Pass-1 exact sizing of the string pool (was `file_len + 1024`; now sums actual `len + 1` for each string). | `src/format/gguf/gguf.c` (pass-1 scan)              | `make sanitize` (LeakSan + ASan).                                                  |
| GGUF non-power-of-two alignment                | Reject in `op_close` precondition; would otherwise be used as a bitmask.                                | `src/format/gguf/gguf.c` (`is_pow2` check)          | `make sanitize`.                                                                   |
| GGUF 0-rank or 0-dim tensor                    | Reject as malformed.                                                                                    | `src/format/gguf/gguf.c` (tensor-descriptor loop)   | `make sanitize`.                                                                   |
| GGUF Q4_K / Q8_0 element count not a multiple of block | `tensor_bytes` returns 0; treated as fatal.                                                      | `src/format/gguf/gguf.c` (`tensor_bytes`)           | `make sanitize`.                                                                   |
| **`.eosm` tampered file**                      | Mandatory SHA-256 trailer recomputed on open; mismatch → `EOS_E_FORMAT` before any sections are parsed. | `src/format/eosm/eosm.c` (`eosm_open`)              | `tests/unit/test_runner.c::test_eosm_roundtrip` (tamper subtest).                  |
| **`.eosm` capability bit not supported**       | Required-bit list cross-checked at open; any unsupported → `EOS_E_UNSUPPORTED` before parse.             | `src/format/eosm/eosm.c`                            | (covered by reader unit tests; failure path manually exercised)                   |
| **`.eosm` header malformed (magic / version / alignment / flags)** | Header validated immediately after read: magic, version ≤ 1, alignment power-of-2 in [32, 65536], flags=0, n_groups>0. | `src/format/eosm/eosm.c` (`eosm_open`)             | `make fuzz-eosm` (CI: 30s libFuzzer + ASan + UBSan).                                |
| **`.eosm` overflow on per-tensor offset within a load group** | Two-step bounds check: `local_offset > group->data_bytes` OR `data_bytes > group->data_bytes - local_offset`. | `src/format/eosm/eosm.c` (tensor resolve loop)      | `make fuzz-eosm`.                                                                  |
| **`.eosm` continuous fuzzing**                 | libFuzzer harness + 3-seed corpus; runs 30s on every PR.                                                | `tests/fuzz/fuzz_eosm.c`, `.github/workflows/ci.yml` (`fuzz-eosm` job) | CI per PR.                                                                         |
| **GGUF continuous fuzzing**                    | libFuzzer harness + 3-seed corpus (deterministic builder; no real model file needed). 30s/PR with ASan + UBSan + LeakSan. | `tests/fuzz/fuzz_gguf.c`, `tests/fuzz/seed_gguf_corpus.c`, `.github/workflows/ci.yml` (`fuzz-gguf` job) | CI per PR.                                                                         |
| **Lifecycle bugs across model_open / session_open / close** | `eosllm-cli --smoke` exercises the full lifecycle against an in-memory synthetic GGUF (no file). `make sanitize-cli` runs the smoke under ASan + UBSan + LeakSan with `abort_on_error=1:halt_on_error=1`. CI cell `smoke-cli`. | `tools/eosllm-cli/main.c` (`run_smoke`), `Makefile` (`smoke`, `sanitize-cli`), `.github/workflows/ci.yml` | CI per PR.                                                                         |
| **Kernel-level memory bugs** (unaligned AVX2/NEON loads, OOB on weight buffers, leaks across malloc cycles) | `make sanitize-bench` builds + runs `eosllm-bench` under ASan + UBSan + LeakSan with abort flags. CI cell runs both `--shapes-only` and full bench. | `tools/eosllm-bench/main.c`, `Makefile` (`sanitize-bench`), `.github/workflows/ci.yml` | CI per PR (in `smoke-cli` job).                                                    |
| **Silent perf regression** (kernel rewrite slowing things down)        | `make benchmark` produces a JSON snapshot uploaded as a per-PR artifact. `make bench-diff BASELINE=...` exits non-zero on >threshold regression. Caught its own real bug during introduction (recursive `make` reused stale `.o`, silently producing a non-AVX2 JSON). | `tools/eosllm-bench/main.c`, `tools/bench_diff.py`, `.github/workflows/ci.yml` (`benchmark` job) | Artifact uploaded per PR; manual diff today, automated diff vs `main` is a follow-up. |
| BPE encoder dereferences NULL on bad input     | NULL/length checks at every entry point.                                                                | `src/tokenizer/bpe.c` (`bpe_encode`, `bpe_decode`)  | `tests/unit/test_runner.c::test_bpe_synthetic` (14 checks).                       |
| BPE merge cycle / pathological input pinning CPU | Merge loop strictly decreases `n` each pass; pair-hash lookup is O(1).                                | `src/tokenizer/bpe.c` (`best_merge`, `bpe_encode`)  | Same.                                                                              |
| Module registered after a session is open      | Registry-freeze policy: every `eos_*_register` returns `EOS_E_INVALID_ARG` after first session open.    | `src/core/registry.c`, `src/core/session.c`         | `tests/unit/test_runner.c::test_registry_frozen` (9 checks).                       |
| `eos_init_defaults` double-registers built-ins | Idempotency flag on `eosi_registry_t`; second call returns `EOS_OK` without pushing.                    | `src/core/init.c`                                   | `tests/unit/test_runner.c::test_init_defaults` (table-size invariants).            |
| Uninitialized-memory read producing flaky output | `make determinism` runs the test runner 3× and demands byte-identical stdout.                         | `Makefile` (`determinism` target)                   | CI job `determinism`.                                                              |
| Memory-safety bugs (use-after-free, OOB)       | `make sanitize` builds with `-fsanitize=address,undefined` + `ASAN_OPTIONS=detect_leaks=1`.             | `Makefile`, `.github/workflows/ci.yml`              | CI job `sanitize`.                                                                 |
| SIMD vs scalar drift (silent miscompute)       | Randomized parity stress: Q8_0 within 1e-5, Q4_K within 1e-4 against scalar oracle. AVX2 + NEON share the harness. | `tests/unit/test_runner.c` (parity tests)          | `make EOSLLM_HAVE_KERNEL_AVX2=1 test` and the cross-aarch64 CI cell.              |

## 7. Residual risks (known and accepted for v0.1)

These are **known**. They are listed so a host operator can apply
compensating controls, not because we believe they are unimportant.

1. **GGUF reader is now fuzzed.** `make fuzz-gguf` builds a libFuzzer
   harness over the GGUF parser; seed corpus comes from a deterministic
   builder (no real model file required). CI cell `fuzz-gguf` runs
   30s on every PR with ASan + UBSan + LeakSan. Beyond the 30s/PR
   budget, longer (≥1 hour nightly) runs are still TODO.
2. **Side-channel timing.** All current attention implementations leak
   prompt content via timing of memory accesses, KV-cache prefetch
   patterns, and matmul block schedules. This is not eosllm-specific
   but is not mitigated either. Hosts that need constant-time
   inference must use a different engine.
3. **No per-session memory limit.** The host's `eos_os_alloc` is the
   only backstop. A future revision may add a soft cap in
   `eos_session_params_t`.
4. **Pre-tokenizer mismatch for Llama-3 / tiktoken.** The
   `tokenizer.ggml.pre` field is recorded but the corresponding regex
   pre-tokenizer is not implemented. Encodings for those models will
   not match the reference Python `tokenizers` library; a `WARN` is
   logged. Tracked in Session 1's full close-out.
5. **AVX2 / NEON kernels are bit-exact-vs-scalar tested for `matmul_q4_k`
   only** (within 1e-4 on randomized inputs). NEON parity is exercised
   via the cross-aarch64 CI cell with qemu-user-static; not on the
   developer host. `matmul_q8_0` SIMD paths and `matmul_f32` random
   parity are still pending.
6. **`.eosm` v1 reader/writer is implemented and fuzzed**, but the
   spec covers only what's needed for v0.1. Items deferred to v2 of
   the format (per-load-group SHA-256, compressed metadata, signed
   manifest) are listed at the end of `docs/file_format.md`.

## 8. Reporting vulnerabilities

This is a research-stage project. There is no embargo process and no
CVE assignment. If you find a bug that could corrupt host memory,
open an issue tagged `security` or contact the maintainer directly.
Do **not** include weaponized payloads in public issues until a fix
is in `main`.

## 9. Out of scope for v0.1

- NPU delegate side-channels (no NPU code yet).
- Continuous batching with prompt-mixing across tenants (no batching).
- Speculative decoding side-channels (no speculative path).
- Streaming / async APIs.
- TLS / network model fetch (the library has no network).
- Signature verification of model files.
- ABI for capability sandboxing (e.g. dropping registry entries the
  loaded model doesn't request).

## 10. Sign-off

A maintainer signs off on this document at every tagged release.
Sign-off means: "I have re-walked the table in [§6](#6-mitigations)
and confirmed every mitigation is still present in the code at the
revision being tagged."

| Tag    | Date       | Sign-off (handle)              |
|--------|------------|---------------------------------|
| v0.1.0 | _pending_  | _pending_                       |

---

## MCP surface: hostile-protocol posture (2026-10-07)

eosllm as an on-device inference service exposes an MCP/tool surface. This
threat-model update adopts a **hostile-protocol posture**: the MCP protocol
itself is treated as hostile-by-default, not just any particular
implementation.

Evidence (Oct 6 reporting): the ClawSecure AI Agent Threat Report Vol 1
(Sept 24, reported Oct 6) documents cracked Linear MCP, Notion, and Dropbox
Dash integrations and concludes the gap sits **in the MCP protocol itself**,
not in vendor bugs. The chain-of-trust story is concrete: MCP servers hold
credentials for every agent, agents trust each other by default, and one
compromised agent becomes a launchpad into the network.

Requirements this posture imposes on the eosllm MCP surface:

1. **Allowlisted servers** — no server is loaded unless explicitly allowed.
2. **Pinned versions** — the allowed set pins exact server versions; upgrades
   are reviewed changes, not floating tags.
3. **Credential isolation from servers** — servers never hold credentials
   they do not need; credentials live outside the server trust boundary.
4. **Upstream-response redaction** — responses from MCP servers are
   scrubbed before they reach the agent context (see also the eIPC
   upstream-response redaction requirement, same date).
5. **No implicit intra-network trust** — agents and servers on the same
   network do not trust each other by default; every crossing is
   authenticated.

Governance note: MCP is now a Linux Foundation project — shared governance
with Zephyr. Track normative security requirements as they develop there;
this posture stands until the spec carries them.
