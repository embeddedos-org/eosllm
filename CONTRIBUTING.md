# Contributing to eosllm

Thanks for your interest. eosllm has strict rules to keep the engine small,
deterministic, and portable. Please read these before opening a PR.

## Non-negotiable rules

1. **C99 only in the engine.** No C++, no compiler-specific extensions in
   public headers. `src/` may use `__attribute__` etc. only behind a guard
   (`EOS_INLINE`, `EOS_LIKELY`, …) defined in `include/eosllm/eosllm.h`.
2. **Zero allocations in the hot path.** All inference memory comes from a
   preallocated arena sized at `eos_session_open`. Use `eos_os_alloc` only at
   init/teardown, never inside `eos_session_step`.
3. **No direct OS calls.** The core never calls `malloc`, `pthread_*`, `fopen`,
   `time`, etc. Use the `eos_os_*` shims declared in `include/eosllm/os.h`.
4. **Modules are translation units behind a vtable.** Adding or removing a
   backend / quant / modality / tokenizer / OS shim must not require touching
   any other module. See `docs/architecture.md`.
5. **Compile-time gating.** Every optional module sits behind an
   `EOSLLM_HAVE_*` flag so the binary stays minimal.
6. **Determinism first.** Same input + same seed + same build flags ⇒
   bit-identical output. PRs that introduce nondeterminism are rejected.
7. **Oracle parity for kernels.** Every accelerated kernel ships a
   randomized-input test against `src/kernels/scalar/`. No exceptions.

## PR checklist

- [ ] `make test` passes locally.
- [ ] No new direct `malloc`/`free`/`pthread`/`fopen` calls in `src/core/` or
      `src/kernels/` or `src/modality/`.
- [ ] No new public symbol added without an entry in `include/eosllm/`.
- [ ] If you added a new module, it is gated by an `EOSLLM_HAVE_*` flag.
- [ ] If you added a new accelerated kernel, you also added an oracle-parity
      test under `tests/unit/`.
- [ ] `CHANGELOG.md` updated under `[Unreleased]`.

## Style

- 4-space indent, no tabs.
- `snake_case` for functions and variables; `EOS_` / `eos_` prefixes for public
  symbols; `EOSI_` / `eosi_` for internal.
- Headers use `#pragma once` plus include guards (some MCU toolchains still
  prefer guards; both is cheap insurance).
- Public functions document ownership in the doc-comment: who allocates, who
  frees, lifetime of returned pointers.

## ABI changes

The public ABI is governed by `docs/abi.md`. Until `1.0.0` we may break it,
but every break must be documented in `CHANGELOG.md` and bump
`EOSLLM_ABI_VERSION` in `include/eosllm/eosllm.h`.
