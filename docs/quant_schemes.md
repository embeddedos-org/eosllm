# Quant Schemes

Each scheme is a translation unit under `src/quant/` and registers with
the engine via `eos_quant_register()`. The scalar reference matmul in
`src/kernels/scalar/` is the always-correct oracle every accelerated
backend is tested against bit-exactly.

## Implemented (Phase 1)

| Scheme | dtype enum     | Block elems | Bytes/block | Status                              |
|--------|----------------|-------------|-------------|-------------------------------------|
| q8_0   | `EOS_DT_Q8_0`  | 32          | 34          | Full: dequant + quant + matmul (GGUF-compat: `f16` scale + 32 `int8`). |
| q4_k   | `EOS_DT_Q4_K`  | 256         | 144         | Dequant + matmul implemented (GGUF-compat super-block). Quantize is convert-tool-only. |

## Implemented (Phase 2)

| Scheme | dtype enum     | Block elems | Bytes/block | Status                              |
|--------|----------------|-------------|-------------|-------------------------------------|
| q1_58  | `EOS_DT_Q1_58` | 160         | 36          | Full: BitNet-style ternary {-1,0,+1}, 5 trits/byte (3⁵=243<256), absmean quant. |

## Scaffolded (Phase 2 follow-ups)

- `q2_k` — GGUF 2-bit K-quant; planned implementation mirrors q4_k.
- `mixed` — per-tensor / per-channel scheme dispatch via `.eosm` manifest.
- `calibrated` — AWQ/GPTQ-style applied at convert time; runtime
  multiplies the underlying scheme's dequant by per-group scales.

## Adding a new scheme

1. Create `src/quant/<name>.c` implementing the `eos_quant_vt_t` vtable:
   `layout`, `dequantize`, `quantize`. Gate the file with
   `EOSLLM_HAVE_QUANT_<NAME>`.
2. Add `<name>.c` to `SRC` in the top-level `Makefile` and a default
   flag in `build/config.mk.in`.
3. Add a registration entry point `eosi_quant_<name>_register()` and
   wire it into `src/quant/register_all.c`.
4. Add the dtype to the `EOS_DT_*` enum in `include/eosllm/tensor.h`
   and to the scalar matmul dispatcher in
   `src/kernels/scalar/register.c`.
5. Add a roundtrip + oracle-parity test under `tests/unit/`.
6. Bump `EOSLLM_ABI_VERSION` and document in `CHANGELOG.md`.
