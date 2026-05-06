#!/bin/bash
set -euo pipefail
cd /home/spatchava/embeddedos-org/eosllm

echo "=== commit 1/8: build ==="
git add Makefile build/ .gitignore
git commit -m "build: Makefile, build/config.mk.in, .gitignore for v0.1

Top-level Makefile drives the full library + tools + tests + sanitize
+ determinism + fuzz + benchmark + smoke + all-checks pipeline. Feature
flags resolved via build/config.mk.in (overridable on the make command
line, e.g. EOSLLM_HAVE_KERNEL_AVX2=1). HOST_ARCH-aware autoconf
auto-enables AVX2 on x86_64 hosts whose /proc/cpuinfo mentions avx2,
and NEON on aarch64. .gitignore excludes build artifacts, coverage
files, and per-host docs/benchmarks/host.json (whitelists
baseline-*.json so committed perf references survive git clean)."

echo "=== commit 2/8: core ==="
git add include/ src/core/ src/os/ src/util/ src/modality/ src/sched/
git commit -m "core: public C99 ABI + engine core + OS shims + modality/sched stubs

include/eosllm/ is the entire public ABI surface: eosllm.h (status
codes, init, caps, version, eos_free, eos_backend_available,
eos_last_error), model.h (open/close, metadata + tensor accessors),
session.h (open/feed/step/decode), os.h (eos_os_provide), scheduler.h.

src/core/ implements:
  - registry.c: backend / format / tokenizer / scheduler / quant
    registries with freeze-after-session_open semantics.
  - session.c: lifecycle orchestrator with EOSI_LOG_ERROR context at
    each module-failure point (text decoder, tokenizer, scheduler,
    greedy-attach alloc).
  - model.c: lifecycle + metadata accessors with eosi_set_error(NULL)
    at every entry point (the AP contract).
  - error.c: thread-local last-error string (__thread / _Thread_local
    fallback to static for toolchains without TLS).
  - status.c, version.c, caps.c, tensor.c, init.c.

src/os/ has POSIX + Zephyr + FreeRTOS + bare-metal shims behind feature
flags. src/util/sha256.c is the FIPS-180-4 implementation used by the
.eosm trailer. src/modality/ and src/sched/ have register tables +
stubs for the v0.2 scope."

echo "=== commit 3/8: kernels ==="
git add src/kernels/ src/quant/
git commit -m "kernels: scalar + AVX2 + NEON matmul + Q4_K/Q8_0/Q1_58 quantization

Backends register via eos_backend_register() with a runtime probe.
Each backend's matmul vt covers matmul_f32 and matmul_q (dispatch to
quant-specific path). The scalar backend is the correctness baseline
referenced by the parity tests.

src/kernels/avx2/matmul_q4_k.c: per-super-block fused dequant + dot
using _mm256_cvtepu8_epi32 + _mm256_fmsub_ps + _mm256_fmadd_ps. SSE2-
only horizontal reduce (avoids the SSE3 _mm_hadd_ps that gcc 9 won't
compile under bare -mavx2).

src/kernels/neon/matmul_q4_k.c: vmovl_u8 + vcvtq_f32_u32 + vmlaq_f32
mirror of the AVX2 path; aarch64 vaddvq_f32 reduce with ARMv7-A
fallback.

src/quant/ implements Q8_0 (per-32-element f16 scale + s8), Q4_K (256-
element super-block with min/scale + 4-bit nibbles), Q1_58 ternary,
mixed-precision orchestrator, and a calibrated-tables passthrough."

echo "=== commit 4/8: format ==="
git add src/format/ src/tokenizer/
git commit -m "format: GGUF v3 reader (audit-hardened) + .eosm v1 reader/writer/converter + BPE

src/format/gguf/gguf.c: GGUF v3 reader, audit-hardened. Exact pass-1
string-pool sizing, overflow-safe tensor offset checks (rejects offset
> file_size or data extending past EOF), 0-rank/non-power-of-two
alignment/Q4_K-not-mod-256 all rejected. EOSI_LOG_ERROR sites cover
every format-rejection path so eos_last_error() carries human-readable
context.

src/format/eosm/: full v1 native format. eosm.c reader validates the
SHA-256 trailer BEFORE any section parsing; capability bits gate
unsupported features at probe time. writer.c emits two-pass byte-
identical layout. from_model.c is the format-agnostic converter
(eosi_eosm_from_model) that takes caller-supplied (key, kv_type) specs;
ARRAY of STRING and ARRAY of U32 supported (covers GGUF vocab +
token_type passthrough).

src/tokenizer/bpe.c: byte-level BPE with open-addressing FNV-1a vocab
hash, SplitMix64 pair hash for best_merge, special-token longest-match
with first-byte bitmap, byte-fallback set discovery, token_type
classification, EOSI_LOG_ERROR on every reject path."

echo "=== commit 5/8: tests ==="
git add tests/
git commit -m "tests: unit suite (148 checks) + libfuzzer harnesses + smoke scripts

tests/unit/test_runner.c: 148 assertions covering BPE behavior, GGUF
reader positive-path + audit checks, .eosm writer/reader/converter
roundtrip + tampered-trailer rejection, registry freeze, init-defaults
idempotency, randomized matmul Q8_0 stress (24 trials xorshift), Q4_K
all-sub-blocks-active reference (independently derived 5988.0), SHA-256
KAT, AVX2 + NEON kernel parity within 1e-4, eos_free, eos_backend_-
available, eos_last_error contract (slot cleared by successful entry-
point calls), synthetic GGUF positive-path roundtrip.

tests/fuzz/: fuzz_eosm.c + fuzz_gguf.c libfuzzer harnesses with deter-
ministic 3-seed corpus generators (no real model file needed). Make
targets fuzz-eosm / fuzz-gguf / fuzz-corpus / fuzz-corpus-gguf wire
them into CI for 30s runs under ASan + UBSan + LeakSan.

tests/fuzz/run_smoke_san.sh is the documented workaround for running
the cli smoke under sanitizers when PowerShell-via-wsl mangles env-var
heredocs (use a script, not a one-liner). tests/data/.gitignore +
README + tools/fetch_test_models.sh form the model-fetch scaffold for
future TinyLlama/Llama-3 E2E tests."

echo "=== commit 6/8: tools ==="
git add tools/
git commit -m "tools: eosllm-cli + eosllm-bench + eosllm-convert + bench_diff.py

tools/eosllm-cli/main.c: prompt-driven CLI plus three model-file-
independent smoke variants. --smoke runs the in-memory synthetic-
GGUF lifecycle (model_open + session_open + close on a 176-byte
hand-built GGUF stream). --smoke-bad-magic exercises the GGUF
reader's reject path. --last-error is a packager sanity-check that
the TLS + EOSI_LOG_ERROR + eos_last_error wiring linked correctly.

tools/eosllm-bench/main.c: real matmul microbench. Walks every
registered backend whose runtime probe passes; runs matmul_f32 /
_q8_0 / _q4_k at 6 fixed shapes (M=1, N in {1,8,64,128,256,512},
K in {256..4096}); per-shape iteration count auto-tuned to ~50ms;
deterministic xorshift32 inputs; emits JSON with backends_probed
+ records[] for downstream diff tooling. Falls back to noise-fill
of the Q4_K weight buffer when qvt->quantize is unsupported.

tools/eosllm-convert/main.c: C-side from-gguf converter that probes
a fixed Llama-family key table and calls eosi_eosm_from_model. Python
wrapper at tools/eosllm-convert/eosllm_convert/cli.py shells out via
subprocess; finds the binary via env var / repo-relative / PATH.

tools/bench_diff.py: joins two bench JSONs by (backend, op, m, n, k),
prints percent-change table, exits non-zero if any record regresses
beyond threshold (default 10%). Drives make bench-diff."

echo "=== commit 7/8: ci ==="
git add .github/
git commit -m "ci: 9 PR-gating GitHub Actions jobs

.github/workflows/ci.yml gates each PR on:
  1. host-build matrix (ubuntu+macos x gcc+clang x debug+release)
  2. cross-aarch64 (qemu-user-static; CFLAGS_EXTRA so DEFS aren't
     overridden when the cross compiler runs)
  3. sanitize (clang ASan + UBSan + LeakSan over the unit suite)
  4. determinism (3-iteration bit-identical stdout)
  5. fuzz-eosm (30s libfuzzer + ASan against the seed corpus)
  6. fuzz-gguf (30s libfuzzer + ASan)
  7. benchmark (perf snapshot + actions/upload-artifact@v4 with name
     bench-\$GITHUB_SHA)
  8. smoke-cli (make smoke-all + make sanitize-cli + sanitize-bench;
     all three smoke shapes verified including last_error context
     substring assertion)"

echo "=== commit 8/8: docs ==="
git add docs/ CHANGELOG.md CONTRIBUTING.md LICENSE README.md
git commit -m "docs: architecture, abi, threat-model, file-format, benchmarks + CHANGELOG/README

docs/architecture.md: hexagonal layering, registry layout, build
targets table (test, smoke, smoke-all, sanitize variants, benchmark,
bench-diff, all-checks).

docs/abi.md: stability promise (pre-1.0 may break per release), public
function reference, last-error string semantics (every public entry
point clears the slot, so EOS_OK leaves it NULL).

docs/threat_model.md: v0.1 scope, attack surface (5 entry points
including .eosm), mitigations table mapped to source files +
test/fuzz harnesses + CI cells (sanitize, fuzz-eosm, fuzz-gguf,
sanitize-cli, sanitize-bench, smoke-cli, benchmark).

docs/file_format.md: byte-level .eosm v1 spec (header, capability
bits, KV section, tensor manifest, tensor data, SHA-256 trailer).

docs/porting.md: how to add a new backend / quant / format /
tokenizer / scheduler / OS shim. References the registry vt
contracts.

docs/benchmarks.md: workflow doc for make benchmark, bench-diff,
sanitize-bench, the per-PR JSON artifact, and the reference table.

docs/benchmarks/baseline-x86_64.json: Threadripper PRO 3975WX
reference for bench-diff to compare against until the CI runner
seeds its own.

CHANGELOG.md: Keep-a-Changelog format. Comprehensive Added/Changed
sections covering every shipped task.

CONTRIBUTING.md, LICENSE, README.md: meta files for the public repo."

echo ""
echo "=== final state ==="
git log --oneline -10
echo ""
echo "=== pushing to origin/main ==="
git push origin main
