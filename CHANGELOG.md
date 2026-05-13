# Changelog

## [3.0.0] - 2026-05-13

### Production Release — Unified EmbeddedOS-org v3.0.0

This is the synchronized production release across all 18 EmbeddedOS-org repos.

- Refreshed governance: LICENSE, NOTICE, CITATION.cff, SECURITY.md
- CI/CD pipelines hardened: release.yml, book-build.yml, video-build.yml, deploy-pages.yml
- Release artifacts produced for: Linux x64/arm64, macOS x64/arm64, Windows x64, Docker, plus per-repo embedded/mobile/extension targets
- mdBook documentation built and deployed to GitHub Pages
- Promo video rendered and attached as a release asset

All notable changes to eosllm are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to semantic versioning once it reaches `1.0.0`. Until then, every
release may break the ABI; see `docs/abi.md` for the stability promise.

## [Unreleased]

### Added

- **Multi-platform release pipeline** (Phase A, `.github/workflows/release.yml`)
  Replaces the single linux-x86_64 job with parallel matrix builds for
  **linux x86_64**, **linux aarch64** (cross + qemu), **macOS
  universal** (arm64 + x86_64 lipo), and **windows x86_64** (MinGW-w64).
  Each platform's signing/notarization is conditional on the
  corresponding GitHub repo secret (`APPLE_*`, `WINDOWS_CERT_*`,
  `MINISIGN_*`); missing secrets cause the workflow to ship the binary
  unsigned with a clear note in the release body. Adds `src/os/win32.c`
  Win32 OS shim, `src/os/wasm.c` + `src/kernels/wasm/wasm.c` Phase F
  stubs, `build/toolchains/win32-x86_64.mk`, helper scripts under
  `tools/release/` (`build_macos_universal.sh`,
  `sign_and_notarize_macos.sh`, `sign_windows.ps1`), `docs/release.md`
  operator manual, and a Release-signing-keys section in `SECURITY.md`.
  CI also now runs the host-build matrix on `windows-latest` so
  Windows portability bugs surface on every PR.
- **`eosllm-cli --stream-jsonl`** (Phase B, `tools/eosllm-cli/main.c`)
  Emits one JSON-Lines record per token to stdout
  (`{"t":"<text>","i":<id>}`) followed by exactly one terminator
  (`{"done":true,"reason":"...","n_tokens":N,"ms":M,"tok_per_s":T}`).
  Stable wire-protocol contract documented in `docs/cli.md`. The
  VS Code extension and browser extension consume this format.
  `--max-tokens` accepted as an alias for `--n`. `--smoke
  --stream-jsonl` emits a 3-token synthetic stream useful as a
  protocol smoke for client implementers.
  New unit test `test_jsonl_protocol_shape` validates the schema
  (5 checks) so future schema changes break the suite.
- **VS Code extension** (Phase C, `vscode-extension/`)
  TypeScript extension that wraps `eosllm-cli --stream-jsonl` and
  streams tokens into a chat webview. First-launch postinstall
  fetches the platform-matching engine binary from the engine's
  GitHub Releases page. Tag-driven publisher workflow at
  `.github/workflows/release-vscode-extension.yml` (triggers on
  `vscode-vX.Y.Z`); conditionally publishes to the VS Code
  Marketplace + Open VSX (each gated on `VSCE_PAT` / `OVSX_PAT`).
- **`eosllm-server` HTTP/SSE daemon** (Phase D, `tools/eosllm-server/`)
  Pure C99 single-binary daemon, zero deps. Routes:
  `GET /healthz`, `GET /caps` (mirrors `eosllm-cli --caps`),
  `POST /generate` (request → SSE token stream). Loopback-bind by
  default; CORS default-deny with `--allow-origin` allow-list.
  Cross-platform (POSIX + winsock2 wrappers). Wire protocol
  documented as a stable contract in `docs/server.md`. Wired into
  `make tools` and `make server` targets; new `make server-smoke`
  health gate; CI cell `server-smoke` runs it on every PR.
- **Browser extension** (Phase E, `browser-extension/`)
  Chrome MV3 + Firefox WebExtension single codebase (vite +
  `@crxjs/vite-plugin`). Talks to `tools/eosllm-server` on
  `localhost:7777`. Tag-driven publisher workflow at
  `.github/workflows/release-browser-extension.yml` (triggers on
  `browser-vX.Y.Z`); conditionally publishes to Chrome Web Store
  (`chrome-webstore-upload-cli`) and Mozilla AMO (`web-ext sign`).
- **`v1.0.0` ABI freeze candidate** (Phase B, `docs/abi.md`)
  New section enumerates every public symbol intended to be part
  of the v1.0.0 ABI plus the `eosllm-cli --stream-jsonl` and
  `eosllm-server` HTTP/SSE wire protocols as out-of-band tooling
  contracts pinned by the same promise.

### Added (existing entries)

- **AVX2 `matmul_q4_k_int8` fused-int8 path**
  (`src/kernels/avx2/matmul_q4_k_int8.c`): a second AVX2 implementation
  of `matmul_q4_k` that keeps the 4-bit weights packed-then-u8 through
  the inner loop and quantizes the activation to int8 once per
  super-block (cached on the stack, reused across all N output
  columns). Inner dot product uses the `_mm256_maddubs_epi16` +
  `_mm256_madd_epi16` ML primitive pair over u4 × i8 lanes; per-
  sub-block rescale-and-accumulate completes the
  `d*sc*sum(q*a) - dmin*m*sum(a)` identity. Wired through the AVX2
  backend's `matmul_q` dispatcher; the f32-dequant path
  (`eosi_avx2_matmul_q4_k`) stays callable from the unit suite for
  tighter (1e-4) parity coverage and is the small-N fallback (N<4) for
  shapes where the activation block-quantize doesn't amortize. New
  parity test `test_avx2_q4_k_int8_parity` (5e-3 relative tolerance,
  N=4 to exercise the cached-activation path; documented in
  `docs/quant_schemes.md`). Microbench: **+37%–+90% throughput** on
  the q4_k AVX2 records vs the f32-dequant baseline at production
  shapes (n=8…512, k up to 4096); n=1 routes to the fallback path
  with no regression.
- **Public ABI additions**
  - `eos_free(void *)` (`include/eosllm/eosllm.h`): release a buffer
    the engine allocated (e.g. the `.eosm` bytes from
    `eosi_eosm_from_model` returned through `eosllm-convert`). Hosts
    cannot use libc `free` because the engine routes allocations
    through `eos_os_provide`. Tolerates NULL.
  - `eos_backend_available(name)` (`include/eosllm/eosllm.h`): walks
    the registry, runs the backend's probe, returns 1/0. Lets hosts
    ask "is AVX2 actually usable on this CPU" instead of only "was it
    compiled in". Names: `scalar`, `x86_avx2`, `x86_avx512`,
    `arm_neon`, `arm_sve`, `rvv`, `hvx`, `npu`.
  - `eos_last_error()` (`include/eosllm/eosllm.h`): thread-local
    null-terminated string carrying the most recent failing call's
    human-readable context (e.g. `"gguf: tensor has rank 0 or > EOS_MAX_RANK"`,
    `"eosm: SHA-256 trailer mismatch"`). Backed by `__thread` /
    `_Thread_local` where available; static fallback elsewhere.
    Wired into the existing `EOSI_LOG_ERROR(...)` macro so every
    parser error path now sets it. Documented in `docs/abi.md`.
- **`eosllm-bench` real microbenchmark** (`tools/eosllm-bench/main.c`)
  - Walks every registered backend whose runtime probe passes; runs
    `matmul_f32`, `matmul_q8_0`, `matmul_q4_k` at 6 fixed shapes
    (M=1, N∈{1,8,64,128,256,512}, K∈{256…4096}); per-shape iteration
    count auto-tuned to ~50 ms; deterministic xorshift PRNG inputs;
    times via `eosi_now_ns`.
  - Emits JSON: `{ library_version, abi_version, backends_probed,
    records[] }`.
  - `--shapes-only` flag for downstream tools.
  - Builds `Q4_K` weight buffer with deterministic noise when
    `qvt->quantize` is unsupported (Q4_K case).
- **`make benchmark` shortcut**
  - Auto-detects host ISA via `uname -m`; sets
    `EOSLLM_HAVE_KERNEL_AVX2=1` on x86_64, `EOSLLM_HAVE_KERNEL_NEON=1`
    on aarch64. Does a `make clean` first so the autoconf flags
    actually take effect. Writes JSON to `docs/benchmarks/host.json`.
  - **Bug caught and fixed during writing**: original target reused
    stale `.o` from `make test`, silently producing JSON without
    AVX2 records. The diff tool exposed it on the very first run.
- **`tools/bench_diff.py` + `make bench-diff`**
  - Joins two bench JSONs by `(backend, op, m, n, k)`, prints
    percent-change table, exits non-zero if a record's
    `giga_ops_per_s` regresses by more than threshold (default 10%).
  - `[NEW]` / `[REMOVED]` reported but not failing.
  - `BASELINE=...` overridable; default
    `docs/benchmarks/baseline-$(HOST_ARCH).json`.
  - `.gitignore` whitelist `baseline-*.json` so committed baselines
    survive `git clean`.
- **`make sanitize-bench`** — builds + runs `eosllm-bench` under
  ASan + UBSan + LeakSan with `abort_on_error=1:halt_on_error=1`.
  Catches kernel-level memory bugs (unaligned loads in AVX2/NEON
  intrinsics, OOB reads on weight buffers, leaks across the bench's
  own malloc cycle).
- **`eosllm-cli --smoke`** (`tools/eosllm-cli/main.c`)
  - Builds a minimal valid GGUF v3 in a 1 KB static buffer, opens it
    via the registered `gguf` format vt over an in-memory
    `eos_stream_t`, walks tensors + metadata, attempts
    `eos_session_open` (controlled `EOS_E_NOT_FOUND` because no
    tokenizer metadata), closes the model, opens a NULL-model
    session, exercises `feed`/`step` (controlled `EOS_E_UNSUPPORTED`),
    closes. **No model file required.**
  - Reports each lifecycle step's outcome to stdout. Exits 0 if every
    happy path completes.
  - Closes the silent gap that the session orchestrator had no
    end-to-end test today (only the NULL-model smoke).
- **`make smoke`** — builds CLI + runs `--smoke`. One-shot health
  check after a `session.c` / `model.c` / format-vt change.
- **`make sanitize-cli`** — smoke under ASan + UBSan + LeakSan.
  Catches lifecycle bugs (use-after-free across
  `model_close` / `session_close`, leaks in the tokenizer-open
  failure path, OOB on metadata accessors).
- **GGUF libFuzzer harness + seed corpus**
  - `tests/fuzz/fuzz_gguf.c` mirrors `fuzz_eosm.c`: in-memory stream
    + format-vt's probe + open + tensor walk + close.
  - `tests/fuzz/seed_gguf_corpus.c` emits 3 minimally-different
    valid v3 files (minimal-f32 / 2-kv-f32-2d / q8_0 quant) using the
    same byte-emission helpers as the unit-test synthetic builder.
    No real model file needed.
  - `make fuzz-gguf`, `make fuzz-corpus-gguf` Makefile targets.
- **Synthetic GGUF v3 positive-path test** (`tests/unit/test_runner.c`)
  - Hand-built GGUF byte stream (header + 2 KVs + 1 tensor + aligned
    data) fed through the registered format vt; asserts metadata +
    tensor body bytes roundtrip.
  - Closes the gap that GGUF reader had zero positive-path tests
    (only audit-style negative coverage under `make sanitize`).
- **ARRAY KV passthrough in `.eosm` converter**
  - `eosi_eosm_from_model` now copies STRING and U32 ARRAY metadata
    via `eos_model_meta_arr_str_*` / `_u32_*`. Per-spec scratch
    buffers freed on cleanup.
  - Closes the GGUF-vocab gap: `tokenizer.ggml.tokens` (STRING array)
    and `tokenizer.ggml.token_type` (U32 array) now round-trip through
    GGUF→`.eosm` lossless conversion.
- **C-side `eosllm-convert from-gguf` tool**
  (`tools/eosllm-convert/main.c`): reads a GGUF, probes a fixed
  Llama-family key table for what's present, infers capability bits
  from observed dtypes, calls `eosi_eosm_from_model`, writes the
  output. Now uses `eos_free` for the engine-allocated buffer instead
  of leaking it.
- **Python CLI wrapper for `from-gguf`**
  (`tools/eosllm-convert/eosllm_convert/cli.py`): rewritten to find
  the C binary via `$EOSLLM_CONVERT_BIN` / repo-relative path / PATH,
  shells out via `subprocess.run`. Phase-2 stub messaging removed.
- **CI matrix expanded to 9 PR-gating jobs**:
  host-build (matrix), cross-aarch64, sanitize, determinism,
  fuzz-eosm (30 s), **fuzz-gguf (30 s)**, **benchmark (perf snapshot
  + JSON artifact)**, **smoke-cli (lifecycle + sanitize-cli +
  sanitize-bench)**.
- **`docs/benchmarks.md`** — rewritten to document `make benchmark`,
  `make bench-diff`, `make sanitize-bench`, the CI workflow, and a
  Threadripper PRO informational reference table.
- **`tests/fuzz/run_smoke_san.sh`** — bash script wrapper for
  running `--smoke` under sanitizers. Exists because PowerShell-
  through-`wsl.exe` mangles env-var heredocs; documented as a
  reusable workaround for sanitize+abort_on_error invocations.
  - Byte-level spec frozen in `docs/file_format.md` (header, capability
    bits, KV section incl. new `BLOB` type, tensor manifest with
    `quant_scheme_id` + `calibration_table_idx` + `load_group_id`,
    calibration tables, load-group index, mandatory SHA-256 trailer).
  - Full reader (`src/format/eosm/eosm.c`): trailer-verify-first,
    capability-bits gate, two-pass parse with exact-sized string pool,
    overflow-safe tensor offset bounds, alignment / rank / dim
    validation, per-key error logging.
  - Full writer (`src/format/eosm/writer.c`): two-pass layout
    computation, per-tensor offsets within their load group, SHA-256
    trailer.
  - Model-agnostic converter (`src/format/eosm/from_model.c`): copies
    tensors via `eos_model_*` accessors and metadata per a caller-
    supplied (key, kv-type) spec list. Supports STRING / U64 / U32 /
    F32 + ARRAY of STRING (vocab) and ARRAY of U32 (token_type).
  - C CLI: `tools/eosllm-convert/eosllm-convert from-gguf <in> -o <out>`
    with a Llama-family key table and dtype-inferred capability bits.
  - Python CLI: `tools/eosllm-convert/eosllm_convert/cli.py from-gguf`
    locates the C binary (env, repo-relative, PATH) and shells out.
- **AVX2 fused `matmul_q4_k`** (`src/kernels/avx2/matmul_q4_k.c`):
  per-super-block `_mm256_cvtepu8_epi32` + `_mm256_fmsub_ps` dequant,
  `_mm256_fmadd_ps` accumulator, manual SSE2-only horizontal reduce.
  Wired into the AVX2 backend's `matmul_q` slot; Q8_0 + others fall
  back to scalar.
- **NEON fused `matmul_q4_k`** (`src/kernels/neon/matmul_q4_k.c`):
  structural mirror of the AVX2 path using `vmovl_u8` / `vcvtq_f32_u32`
  / `vmlaq_f32`. aarch64 reduce via `vaddvq_f32`; ARMv7-A fallback
  via pairwise add.
- **SHA-256 utility** (`src/util/sha256.[ch]`): zero-dep FIPS 180-4
  implementation. KAT-verified for `""` and `"abc"` (independently
  confirmed against `sha256sum`).
- **Registry-freeze thread-safety policy**: `eosi_registry_t.frozen`
  flips on first `eos_session_open`; subsequent `eos_*_register` and
  `eos_os_provide` return `EOS_E_INVALID_ARG`. Documented in
  `docs/abi.md`. Enforced in `src/core/registry.c`.
- **`make sanitize`** target: `-fsanitize=address,undefined` with
  `ASAN_OPTIONS=detect_leaks=1`. CI cell runs on every PR.
- **`make determinism`** target: 3-iteration byte-identical stdout
  check. CI cell runs on every PR.
- **libFuzzer harness for the `.eosm` reader** (`tests/fuzz/fuzz_eosm.c`)
  + 3-seed corpus generator (`tests/fuzz/seed_eosm_corpus.c`). New
  `make fuzz-corpus` and `make fuzz-eosm` targets. CI cell runs the
  harness for 30s with ASan + UBSan against the seed corpus on every PR.
- **Randomized matmul parity stress** (`tests/unit/test_runner.c`):
  24-trial Q8_0 dequant-vs-fused parity within 1e-5; Q4_K
  all-sub-blocks-active block matmul parity within 1e-4.
- **Threat model** (`docs/threat_model.md`): 10-section v0.1 document
  with mitigations mapped to source files and tests.
- **`.gitignore`** and `tests/data/README.md` for the gitignored model
  weights directory + `tools/fetch_test_models.sh` (idempotent
  HuggingFace fetcher with SHA-256 verification).

### Changed

- **BPE tokenizer**: replaced O(V) linear vocab lookup with FNV-1a
  open-addressing hash; added pair-hash for `best_merge` (O(1)
  per-pair instead of O(M)). Added special-token longest-match scan
  with a 256-entry first-byte bitmap. Honors `tokenizer.ggml.pre`
  metadata (logs WARN for unimplemented schemes). Auto-discovers
  special tokens from `tokenizer.ggml.token_type`.
- **GGUF reader audit**: replaced over-reserved string pool sizing
  with exact pass-1 byte accounting; overflow-safe tensor
  offset/data_bytes bounds; rejects 0-rank tensors, non-power-of-two
  alignment, Q4_K/Q8_0 element counts not divisible by block size.
  Per-error `EOSI_LOG_ERROR` messages.
- **`eos_init_defaults`**: now truly idempotent — second call returns
  EOS_OK without re-pushing into the registry tables. Test asserts
  table sizes don't grow.
- **Build system**: introduced `CFLAGS_EXTRA` / `LDFLAGS_EXTRA` that
  always append (via `override`) so cross builds keep the
  auto-generated `-DEOSLLM_HAVE_*` defines. Cross-aarch64 CI cell
  fixed to use these.
- **CI matrix**: `.github/workflows/ci.yml` now has `sanitize`,
  `determinism`, and `fuzz-eosm` jobs in addition to host-build and
  cross-aarch64.
- **`docs/abi.md`**, **`docs/architecture.md`**, **`docs/porting.md`**,
  **`docs/file_format.md`**: audited for accuracy vs shipped code,
  6 wrong build-flag defaults fixed in `architecture.md`, sanitize +
  determinism + register-freeze sections added.

### Fixed

- BPE tokenizer used linear vocab lookup; on Llama-3 (~128k vocab)
  this was the dominant encode cost. Open-addressing hash + pair-hash
  fix it.
- GGUF reader allocated `file_len + 1024` bytes for the string pool,
  over-reserving hundreds of MB on multi-GB files. Pass-1 exact
  sizing fixes it.
- Tensor offset bounds check could overflow on hostile input
  (`(size_t)offset + data_bytes` could wrap). Replaced with
  overflow-safe pair of comparisons.
- `eos_init_defaults` double-registered every built-in module on
  second call. The library's existing test only checked the return
  value; the new test snapshots all 6 table sizes and asserts they
  don't grow.
- Cross-aarch64 CI cell overrode `CFLAGS` entirely, losing every
  `-DEOSLLM_HAVE_*` define and silently building a non-functional
  library. Use `CFLAGS_EXTRA` instead.
- `EOSLLM_HAVE_KERNEL_AVX2=1` build was broken on gcc 9: `_mm_hadd_ps`
  needed SSE3 explicitly; replaced with an SSE2-only horizontal
  reduce. Same latent bug existed in `avx2_matmul_f32` (never
  exercised because AVX2 wasn't on by default).
- `Makefile` AVX2 conditional used plain `+=` which is a no-op when
  `CFLAGS` is overridden on the command line. Switched to
  `override CFLAGS +=`.

### Status matrix

| Phase | Module                                     | State           |
|-------|--------------------------------------------|-----------------|
| 0     | scaffold, public ABI, scalar oracle, POSIX | implemented     |
| 1     | f16↔f32 helpers                            | implemented     |
| 1     | GGUF v3 read-only loader                   | implemented (audit-hardened: bounds, sizing, alignment) |
| 1     | q8_0 quant scheme (GGUF-compat)            | implemented; randomized parity vs scalar |
| 1     | q4_k quant scheme (GGUF-compat)            | dequant + matmul (scalar + AVX2 + NEON); quantize is convert-tool-only |
| 1     | Llama-class text decoder                   | implemented     |
| 1     | byte-level BPE tokenizer                   | implemented (hash + special-tokens; regex pre-tokenizer still TODO for Llama-3) |
| 1     | greedy scheduler                           | implemented     |
| 1     | `eosllm-cli` chat / generate               | implemented (links and runs; e2e correctness vs `llama.cpp` requires a user-supplied model) |
| 2     | q1_58 ternary quant                        | implemented     |
| 2     | q2_k quant scheme                          | scaffold        |
| 2     | mixed-precision quant scheme               | scaffold        |
| 2     | calibrated (AWQ-style) quant scheme        | scaffold        |
| 2     | `.eosm` v1 format reader/writer            | **implemented** (header, KV inc. BLOB, manifest, calibration, load groups, SHA-256 trailer) |
| 2     | `eosllm-convert`                           | **implemented** (C CLI from-gguf; Python wrapper shells out; from-hf still scaffold) |
| 2     | `eosllm-quant-lab`                         | scaffold        |
| 3     | vision modality (ViT / SigLIP)             | scaffold        |
| 3     | audio modality (Whisper-class)             | scaffold        |
| 3     | cross-modal fusion                         | scaffold        |
| 4     | AVX2 backend                               | matmul_f32 + matmul_q4_k fused (Q8_0 → scalar) |
| 4     | NEON backend                               | matmul_f32 + matmul_q4_k fused (Q8_0 → scalar) |
| 4     | AVX-512 / SVE / RVV / HVX / NPU backends   | scaffold        |
| 4     | deadline scheduler                         | scaffold        |
| 4     | Zephyr / FreeRTOS / bare-metal OS shims    | scaffold        |
| 5     | continuous-batched scheduler               | scaffold        |
| 5     | paged KV cache                             | scaffold        |
| 5     | speculative decoding                       | scaffold        |
| —     | registry-freeze thread-safety policy       | implemented + tested |
| —     | sanitize / determinism / fuzz CI cells     | implemented (PR-gating) |
| —     | SHA-256 utility (FIPS 180-4)               | implemented + KAT-verified |
| —     | threat model                               | drafted (`docs/threat_model.md`) |

### Verification (host build)

- **126/126** unit checks pass on the default build (`make test`).
- **128/128** with `make EOSLLM_HAVE_KERNEL_AVX2=1 test` (adds the
  AVX2 q4_k parity case; agrees with scalar within 1e-4).
- `make sanitize` is clean (ASan + UBSan + LeakSan).
- `make determinism` reports byte-identical stdout across 3 runs on
  both default and AVX2 builds.
- `make fuzz-corpus` produces 3 valid `.eosm` seed files.
- CI: 5 jobs (host-build matrix, cross-aarch64, sanitize, determinism,
  fuzz-eosm) gate every PR.
- Tools: `eosllm-cli`, `eosllm-bench`, `eosllm-convert` all build and
  run their `--help`.

### Known gaps (called out for honesty)

- **Tokenizer pre-tokenization** for Llama-3 / tiktoken still missing
  (the regex DFA + Unicode tables are not yet shipped). Encodings for
  those models will not be bit-exact vs reference; a `WARN` is logged.
- **q4_k quantize** still convert-tool-only (Python).
- **GGUF reader has no continuous fuzzing yet**: the libfuzzer
  harness covers the `.eosm` reader; GGUF needs its own harness +
  corpus seeded from a real GGUF (blocked on a model file in `tests/data/`).
- **End-to-end `eosllm-cli` vs `llama.cpp`** still not in CI; needs
  a user-supplied GGUF.
- **NEON `matmul_q4_k` not exercised on this developer host** (no
  aarch64 toolchain locally). The cross-aarch64 CI cell will catch
  build / parity regressions via qemu-user-static.
- **AVX-512 / SVE / RVV / HVX / NPU**: still scaffolds.
- **Calibration tables** in `.eosm` (`AWQ_PER_GROUP_SCALES_F16` etc.)
  pass through the reader/writer as opaque blobs; no quant scheme
  consumes them yet.
