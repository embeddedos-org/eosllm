# eosllm

**Edge-first multi-modal inference engine in pure C99.**

eosllm is an inference runtime designed to run vision, audio, and text models on
the same engine — from server-class CPUs down to MCUs and bare-metal targets. It
is written in standards-conformant C99 with hand-written assembly kernels per
ISA. There is no C++ in the core.

## Goals (vs. `llama.cpp`)

1. **Multi-modal as first-class.** Vision + audio + text share one runtime, one
   ABI, one model file format.
2. **Better quantization.** Sub-2-bit (BitNet 1.58 class), mixed precision, and
   calibrated quant schemes embedded in the model file.
3. **Embedded / edge deployment.** Tiny static binary, deterministic memory
   (zero allocations in the hot path), real-time scheduling, and OS hooks for
   POSIX, Zephyr, FreeRTOS, and bare-metal.

## Status

Phases 0–5 have been wired up (see `CHANGELOG.md` for the full
implemented-vs-scaffolded matrix). What's actually working today:

- **Phase 0** — full scaffold + scalar oracle kernels + POSIX shim + CI.
- **Phase 1** — GGUF v3 read-only loader, q8_0 + q4_k quant schemes,
  Llama-class transformer text decoder (GQA + RoPE + SwiGLU + KV cache),
  byte-level BPE tokenizer, greedy scheduler, `eosllm-cli`.
- **Phase 2** — q1.58 ternary BitNet quant (real). `.eosm` format,
  AWQ-style calibration, and the Python `eosllm-convert` /
  `eosllm-quant-lab` tools are scaffolded.
- **Phase 4** — AVX2 (FMA) and NEON `matmul_f32` backends. Other ISAs
  and edge OS shims are scaffolded.
- **Phases 3, 5** — multi-modal (vision/audio/fusion) and throughput
  features (continuous batching, paged KV, speculative) are scaffolded
  module skeletons.

`make test` runs 62 unit checks (including bit-exact oracle parity for
quant-schemes and SIMD backends). End-to-end `eosllm-cli` against a real
Llama-3 GGUF is the user's first integration test — we don't ship a
model.

## Building

```
make            # host build, default features (scalar + posix + gguf + q8_0/q4_k/q1_58 + bpe + text + greedy)
make test       # build and run the unit test runner (62 checks)
make tools      # build eosllm-cli + eosllm-bench
make config     # show resolved feature flags
make EOSLLM_HAVE_KERNEL_AVX2=1 BUILD=release test
```

Per-feature builds are controlled by `EOSLLM_HAVE_*` flags in
`build/config.mk.in`. See `docs/architecture.md` for the full list.

## License

MIT — see `LICENSE`.

## Layout

```
include/eosllm/   public C99 ABI (the only headers users include)
src/core/         session lifecycle, graph executor, KV cache
src/kernels/      one subdir per ISA; scalar/ is the always-built oracle
src/quant/        one .c per quant scheme
src/modality/     text/, vision/, audio/, fusion.c
src/os/           one .c per target (posix today; zephyr/freertos/baremetal later)
src/format/       model file readers (eosm native; gguf read-only for bring-up)
src/sched/        scheduling policies (greedy, deadline, batched, …)
src/tokenizer/    BPE, sentencepiece-compat, tiktoken-compat
tools/            CLI, converter, bench, quant-lab
tests/            unit/, golden/, fuzz/, targets/
docs/             architecture, ABI, file format, quant schemes, porting
```

## Contributing

See `CONTRIBUTING.md`.
