# Porting eosllm to a new ISA, OS, or NPU

The engine's "everything independent" rule means each port is one new
translation unit plus a few build-flag entries. You never modify the
core to add a backend.

## Port a new ISA backend

1. Create `src/kernels/<isa>/<isa>.c`. Implement an `eos_backend_vt_t`
   with `name`, `priority`, optional `probe`, and a partial `ops`
   table. Any `NULL` op falls back to scalar. Gate the file with
   `EOSLLM_HAVE_KERNEL_<ISA>`.
2. Add an entry to `src/kernels/kernels_internal.h` and register from
   `src/kernels/register_all.c`.
3. Add `<isa>` to the `SRC`, `FLAGS`, and ISA-specific compile-flag
   sections of the top-level `Makefile`.
4. Add a default flag in `build/config.mk.in`.
5. Add an oracle-parity test under `tests/unit/`: feed random inputs
   to your kernel and to `eosi_scalar_*`, assert bit-equality.

See `src/kernels/avx2/avx2.c` and `src/kernels/neon/neon.c` for the
template.

## Port a new OS shim

1. Create `src/os/<os>.c` implementing the `eos_os_shim_t` function
   pointers plus an `eosi_os_use_<os>()` registration function. Gate
   with `EOSLLM_HAVE_<OS>`.
2. Add to `src/os/os_internal.h` and to the `SRC` list in the
   `Makefile`.
3. The hot path **must not** call any platform API not declared in
   `eos_os_shim_t` — that is the porting contract.

## Port a new NPU delegate

NPU "backends" don't supply per-op kernels; they supply a whole
sub-graph delegate. The Phase 4 `src/kernels/npu/` skeleton is the
template — once finished it will expose a single op,
`run_subgraph(prepacked_blob, inputs, outputs)`, and the session's
graph builder will pack the matching layers ahead of time.

## CI matrix

`.github/workflows/ci.yml` holds the matrix. New ports should add a
job that:

- builds `lib` with the port flag enabled;
- runs `make test` (under qemu / a simulator if needed);
- runs `make sanitize` if the port targets a sanitizer-capable
  toolchain (i.e. anything that ships clang / gcc on Linux or macOS);
- records stripped-binary size for the canonical build.

When passing extra compile or link flags to a cross-build, prefer
`CFLAGS_EXTRA=...` / `LDFLAGS_EXTRA=...` over overriding `CFLAGS` /
`LDFLAGS` directly. Overriding `CFLAGS` discards the
`-DEOSLLM_HAVE_*` defines the Makefile auto-generates and produces a
non-functional library.
