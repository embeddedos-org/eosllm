# eosllm Architecture

This document describes the static structure of the engine: modules, their
contracts, the file format, and the build flags. Implementation details live
in subsystem-specific docs (`abi.md`, `file_format.md`, `quant_schemes.md`,
`porting.md`).

## Product family

eosllm is the LLM inference runtime of the
[eAI](https://github.com/embeddedos-org/eAI) product family. eAI is the
on-device AI layer of the EmbeddedOS ecosystem (runtime contracts, agent
runtimes, model-format loaders, platform adapters); eosllm is the engine that
runs large language models inside that layer. This grouping is documentation
only: the repository, the library (`libeosllm.a`), the C ABI (`eos_*` symbols),
and the `eosllm-*` tools keep their existing names.

## Design principles (non-negotiable)

- **Zero allocations in the hot path.** All inference memory is taken from a
  preallocated arena sized at `eos_session_open`. `eos_session_step` never
  calls `eos_os_alloc`.
- **C99 only in the core.** No C++. Public headers are MCU-toolchain friendly.
- **Each backend / quant / modality / OS shim is a translation unit** behind a
  stable vtable. Removing one must not require touching another.
- **Compile-time feature flags.** Every optional module is gated by an
  `EOSLLM_HAVE_*` macro so the binary stays minimal.
- **Determinism first.** Same input + same seed + same build flags ⇒
  bit-identical output across every supported ISA.
- **OS abstraction is a header.** The core never calls `malloc`, `pthread_*`,
  or `fopen`; only `eos_os_*` shims.

## Module map

```
                 ┌────────────────────────────────────────────────┐
                 │                eos_session_t                   │
                 │  arenas · KV · scheduler · dispatch tables     │
                 └────────────┬───────────────────────────────────┘
                              │ vtable lookups (built once at open)
   ┌──────────────────────────┼────────────────────────────────────────┐
   │                          │                                        │
┌──┴────┐  ┌──────────┐  ┌────┴─────┐  ┌────────────┐  ┌──────────┐  ┌─┴───────┐
│format │  │ tokenizer│  │ modality │  │  backend   │  │  quant   │  │ sched   │
│ eosm  │  │  bpe     │  │  text    │  │  scalar    │  │  q8_0    │  │ greedy  │
│ gguf  │  │  spm     │  │  vision  │  │  avx2/512  │  │  q4_k    │  │ deadln  │
└───────┘  └──────────┘  │  audio   │  │  neon/sve  │  │  q2_k    │  │ batched │
                         │  fusion  │  │  hvx       │  │  q1_58   │  └─────────┘
                         └──────────┘  │  npu       │  │  mixed   │
                                       └────────────┘  │ calibrtd │
                                                       └──────────┘
                              │
                       ┌──────┴───────┐
                       │   os shim    │
                       │ posix/zephyr │
                       │ freertos/bm  │
                       └──────────────┘
```

Each box is a self-contained directory under `src/` and is selected at compile
time by an `EOSLLM_HAVE_*` flag. The core never `#include`s a module's
internals; it talks to it only through the registration vtable declared in the
matching public header.

## Public ABI surface

All public types and functions live in `include/eosllm/`. The user only ever
includes `<eosllm/eosllm.h>`.

| Header                       | Purpose                                             |
|------------------------------|-----------------------------------------------------|
| `eosllm.h`                   | top-level: session, generate, embed, version macros |
| `model.h`                    | model file open / close / introspect                |
| `tensor.h`                   | opaque tensor handle, dtype enum                    |
| `quant.h`                    | quant scheme registration vtable                    |
| `modality.h`                 | text / vision / audio encoder vtable                |
| `backend.h`                  | kernel backend vtable                               |
| `scheduler.h`                | scheduling policy vtable, deadline API              |
| `os.h`                       | OS shim vtable (provided by host)                   |

## Module registration

Each pluggable module exposes a single registration entry point:

| Module           | Registration symbol         | Header        |
|------------------|-----------------------------|---------------|
| Backend (kernel) | `eos_backend_register`      | `backend.h`   |
| Quant scheme     | `eos_quant_register`        | `quant.h`     |
| Modality         | `eos_modality_register`     | `modality.h`  |
| Tokenizer        | `eos_tokenizer_register`    | `eosllm.h`    |
| OS shim          | `eos_os_provide`            | `os.h`        |
| Scheduler policy | `eos_sched_register`        | `scheduler.h` |
| File format      | `eos_format_register`       | `model.h`     |

`src/core/registry.c` (added in Phase 1) owns the small static tables that
hold these registrations. A build configured with only `scalar + q8_0 + text +
posix + bpe + greedy + eosm` produces the smallest possible binary.

## Build flags

Defined in `build/config.mk.in`. Every optional module corresponds to one
flag; defaults are set so a `make` with no overrides produces a runnable
host build.

| Flag                            | Default | What it enables                                   |
|---------------------------------|---------|---------------------------------------------------|
| `EOSLLM_HAVE_THREADS`           | OFF     | Multi-threaded scheduler & kernel work-stealing.  |
| `EOSLLM_HAVE_POSIX`             | ON\*    | POSIX OS shim (\*auto-on if target is POSIX).      |
| `EOSLLM_HAVE_ZEPHYR`            | OFF     | Zephyr OS shim.                                   |
| `EOSLLM_HAVE_FREERTOS`          | OFF     | FreeRTOS OS shim.                                 |
| `EOSLLM_HAVE_BAREMETAL`         | OFF     | Bare-metal OS shim.                               |
| `EOSLLM_HAVE_KERNEL_SCALAR`     | ON      | Always-on portable C reference (the oracle).      |
| `EOSLLM_HAVE_KERNEL_AVX2`       | OFF     | x86 AVX2 kernels.                                 |
| `EOSLLM_HAVE_KERNEL_AVX512`     | OFF     | x86 AVX-512 kernels.                              |
| `EOSLLM_HAVE_KERNEL_NEON`       | OFF     | ARM NEON kernels.                                 |
| `EOSLLM_HAVE_KERNEL_SVE`        | OFF     | ARM SVE kernels.                                  |
| `EOSLLM_HAVE_KERNEL_RVV`        | OFF     | RISC-V V kernels.                                 |
| `EOSLLM_HAVE_KERNEL_HVX`        | OFF     | Qualcomm Hexagon HVX kernels.                     |
| `EOSLLM_HAVE_KERNEL_NPU`        | OFF     | NPU delegates (CoreML, NNAPI, QNN, EdgeTPU).      |
| `EOSLLM_HAVE_QUANT_Q8_0`        | ON      | 8-bit quant (bring-up baseline).                  |
| `EOSLLM_HAVE_QUANT_Q4_K`        | ON      | 4-bit K-quant.                                    |
| `EOSLLM_HAVE_QUANT_Q2_K`        | OFF     | 2-bit K-quant (scaffolded).                       |
| `EOSLLM_HAVE_QUANT_Q1_58`       | ON      | BitNet-style 1.58-bit ternary.                    |
| `EOSLLM_HAVE_QUANT_MIXED`       | ON      | Per-layer / per-channel mixed precision.          |
| `EOSLLM_HAVE_QUANT_CALIBRATED`  | ON      | AWQ/GPTQ-style calibrated quant.                  |
| `EOSLLM_HAVE_MODALITY_TEXT`     | ON      | Transformer text decoder.                         |
| `EOSLLM_HAVE_MODALITY_VISION`   | OFF     | ViT / SigLIP encoder.                             |
| `EOSLLM_HAVE_MODALITY_AUDIO`    | OFF     | Whisper-style audio encoder.                      |
| `EOSLLM_HAVE_TOKENIZER_BPE`     | ON      | Byte-level BPE.                                   |
| `EOSLLM_HAVE_TOKENIZER_SPM`     | OFF     | SentencePiece-compatible tokenizer.               |
| `EOSLLM_HAVE_FORMAT_EOSM`       | ON      | Native `.eosm` file format reader/writer.         |
| `EOSLLM_HAVE_FORMAT_GGUF`       | ON      | Read-only GGUF for bring-up.                      |
| `EOSLLM_HAVE_SCHED_GREEDY`      | ON      | Single-stream greedy scheduler.                   |
| `EOSLLM_HAVE_SCHED_DEADLINE`    | OFF     | Deadline-driven real-time scheduler.              |
| `EOSLLM_HAVE_SCHED_BATCHED`     | OFF     | Continuous batching with paged KV.                |

## Roadmap

The original phased delivery plan lives in
`~/.llms/plans/eosllm_initial_architecture.plan.md`. Phases 0–5 trace
the foundational architecture; the production-readiness path to v0.1
is tracked separately in
`~/.llms/plans/eosllm_production_roadmap.plan.md`, which decomposes the
remaining work into 15 numbered sessions with per-session approval.

| Phase | Scope                                                            |
|-------|------------------------------------------------------------------|
| 0     | Foundations: scaffold, public ABI, scalar oracle, POSIX, CI      |
| 1     | Text-only end-to-end (Llama 3 / Qwen2) on x86 AVX2 + ARM NEON    |
| 2     | Native `.eosm` format + sub-2-bit quant                          |
| 3     | Multi-modal MVP (Llava + Whisper class)                          |
| 4     | Edge & real-time (Zephyr / FreeRTOS / bare-metal, deadline sched)|
| 5     | Throughput features (continuous batching, paged KV, speculative) |

## Build, sanitize, determinism

| Target              | What it does                                                                          |
|---------------------|---------------------------------------------------------------------------------------|
| `make`              | Builds `libeosllm.a` with the flags from `build/config.mk.in`.                       |
| `make test`         | Builds + runs the unit suite (currently 91 checks).                                  |
| `make sanitize`     | Clean rebuild with `-fsanitize=address,undefined`, `ASAN_OPTIONS=detect_leaks=1`, runs the suite. Bar: zero reports. |
| `make determinism`  | Runs the unit suite three times back-to-back; fails if stdout differs.                |
| `make smoke`        | Builds CLI + runs `--smoke` (in-memory synthetic-GGUF lifecycle).                    |
| `make smoke-all`    | Runs `--smoke`, `--smoke-bad-magic`, `--last-error` in sequence; each must exit 0.   |
| `make sanitize-cli` | Builds CLI under sanitizers + runs `--smoke`.                                        |
| `make sanitize-bench` | Builds bench under sanitizers + runs the full microbench.                          |
| `make benchmark`    | Auto-enables AVX2/NEON for the host arch; runs `eosllm-bench`; writes `docs/benchmarks/host.json`. |
| `make bench-diff`   | Runs `make benchmark` + diffs against `docs/benchmarks/baseline-$(arch).json`. Fails on >10% per-shape regression. |
| `make all-checks`   | Umbrella: `test` + `smoke-all` + `determinism` + `sanitize` + `sanitize-cli` + `sanitize-bench` + `bench-diff` (when a baseline exists). One command for "is the tree healthy". ~5 min wall-time on a modern x86. |
| `make tools`        | Builds `eosllm-cli` + `eosllm-bench` + `eosllm-convert`.                             |
| `make config`       | Prints the resolved feature-flag values (useful for debugging build issues).         |

For cross builds, pass `CFLAGS_EXTRA=...` / `LDFLAGS_EXTRA=...` rather
than overriding `CFLAGS` / `LDFLAGS` outright — the latter discards
the auto-generated `-DEOSLLM_HAVE_*` defines.

See `docs/threat_model.md` for the security contract and how the
sanitize / determinism targets fit into it.
