# eosllm ABI Contract

This document specifies the C99 ABI exposed by `include/eosllm/`. It is the
contract eosllm offers to host applications and to module authors.

## Versioning

`include/eosllm/eosllm.h` defines:

```c
#define EOSLLM_VERSION_MAJOR 0
#define EOSLLM_VERSION_MINOR 1
#define EOSLLM_VERSION_PATCH 0

/* Bumped on every breaking ABI change while major == 0. */
#define EOSLLM_ABI_VERSION   1
```

`eos_abi_version(void)` returns `EOSLLM_ABI_VERSION` at runtime so a host that
links against a shared build can refuse a mismatched library.

While the major version is `0`, every release may break the ABI. Each break
bumps `EOSLLM_ABI_VERSION` and is documented in `CHANGELOG.md`.

After `1.0.0`:

- **Patch** releases (`x.y.Z`) — bug fixes; ABI compatible.
- **Minor** releases (`x.Y.0`) — additive only (new functions, new enum
  values appended at the end). Old code keeps working.
- **Major** releases (`X.0.0`) — may break the ABI; bumps `EOSLLM_ABI_VERSION`.

## Symbol naming

| Prefix      | Meaning                                                 |
|-------------|---------------------------------------------------------|
| `EOS_` `eos_` | Public — declared in `include/eosllm/`. Stable per the policy above. |
| `EOSI_` `eosi_` | Internal — visible only within `src/`. Never call from a host. |

Public symbols never start with `_` and never collide with the C standard
library namespace.

## Opaque types

All session, model, tensor, and registry handles are opaque pointers. The host
only ever sees `typedef struct eos_session eos_session_t;` etc. Layout may
change between releases without bumping the major version, provided the ABI
itself does not change.

## Error handling

Every entry point returns an `eos_status_t` enum. The library never aborts,
never calls `exit`, and never throws (no longjmp from the core). Out-parameters
are written only on `EOS_OK`.

```c
typedef enum {
    EOS_OK = 0,
    EOS_E_INVALID_ARG,
    EOS_E_OUT_OF_MEMORY,
    EOS_E_UNSUPPORTED,        /* requested capability not compiled in */
    EOS_E_FORMAT,             /* malformed model file */
    EOS_E_VERSION_MISMATCH,
    EOS_E_NOT_FOUND,
    EOS_E_DEADLINE,           /* eos_session_step ran out of time */
    EOS_E_IO,                 /* host file/stream returned an error */
    EOS_E_INTERNAL            /* invariant violation; report a bug */
} eos_status_t;

const char *eos_status_str(eos_status_t s);
```

New error codes are appended at the end. Hosts must treat unknown codes as
`EOS_E_INTERNAL`.

- **Compile-time vs runtime backend availability**

  ```c
  int eos_backend_available(const char *name);  /* per-backend probe */
  ```

  And via `eos_caps()`:

  ```c
  eos_caps_t c;
  eos_caps(&c);
  if (c.have_k_avx2 && (c.runtime_bits & EOS_CAPS_RT_AVX2)) {
      /* AVX2 was compiled in AND the host CPU's probe accepted it. */
  }
  ```

  `runtime_bits` carries one bit per backend whose runtime probe
  passed at the time of the `eos_caps()` call. Masks: `EOS_CAPS_RT_AVX2`,
  `EOS_CAPS_RT_AVX512`, `EOS_CAPS_RT_NEON`, `EOS_CAPS_RT_SVE`,
  `EOS_CAPS_RT_RVV`, `EOS_CAPS_RT_HVX`, `EOS_CAPS_RT_NPU`. A bit
  set in `runtime_bits` always implies the matching `have_k_*`
  compile-time bit is also set; the converse isn't guaranteed (the
  CPU may lack a feature the binary supports).

### Last-error string

For human-readable context beyond the enum, call `eos_last_error()`:

```c
const char *eos_last_error(void);
```

Returns a thread-local null-terminated message describing why the most
recent failing call failed (e.g. `"gguf: tensor has rank 0 or > EOS_MAX_RANK"`,
`"eosm: SHA-256 trailer mismatch"`), or `NULL` if no error has been
recorded since the last successful entry-point call. **Every public
entry point** — lifecycle (`eos_model_open`, `eos_session_open`,
`eos_session_feed`, `eos_session_step`, `eos_session_decode`),
metadata accessors (`eos_model_meta_str` / `_u64` / `_f32` /
`_arr_*_count` / `_arr_*_load`), and tensor accessors
(`eos_model_num_tensors`, `eos_model_tensor`, `eos_model_tensor_by_name`)
— clears the slot at entry, so **after `EOS_OK` (or a non-NULL pointer
return for `_str` / `_tensor`) the slot is always `NULL`** (not stale
context from an earlier failing call). Pointer contents are valid
until the next `eos_*` call on the same thread; copy the bytes if
you need them later.

The buffer is thread-local on gcc/clang/C11; on toolchains without
TLS support it is a single static buffer (documented on a per-port
basis).
`EOSI_LOG_ERROR(...)` macro, so hosts that don't install a log shim
via `eos_os_provide` can still surface "what went wrong".

## Memory ownership

- The library never frees memory it did not allocate via `eos_os_alloc`.
- The host never frees memory returned by the library; instead it calls the
  matching `eos_*_close` / `eos_*_free` entry point.
- Strings returned by the library are valid until the corresponding handle is
  closed.

## Thread safety

The default build (`EOSLLM_HAVE_THREADS=0`) is single-threaded. With threading
enabled, the rules are:

- An `eos_session_t` is **not** thread-safe. One session per thread, or
  external mutual exclusion.
- **Module registration is one-shot.** All `eos_*_register` calls (and
  `eos_os_provide`) must complete before the first `eos_session_open`.
  After that, the registry is **frozen**: every subsequent register call
  returns `EOS_E_INVALID_ARG` and leaves the registry untouched. This
  lets sessions read dispatch tables without locks. The behavior is
  enforced by `src/core/registry.c` and exercised by the
  `test_registry_frozen` checks in `tests/unit/test_runner.c`.
- `eos_init_defaults()` is idempotent. The first call registers every
  built-in module; subsequent calls return `EOS_OK` without re-pushing.
- The OS shim functions provided via `eos_os_provide` must themselves be
  thread-safe if `EOSLLM_HAVE_THREADS` is on.

## Sanitizer & determinism harnesses

Two Makefile targets are part of the v0.1 release contract; both run on
every PR:

- `make sanitize` — rebuild with `-fsanitize=address,undefined` and
  `ASAN_OPTIONS=detect_leaks=1`, then run the unit suite. Zero reports
  is the bar.
- `make determinism` — run the unit suite three times back-to-back and
  confirm byte-identical stdout. Catches uninitialized-memory reads and
  unstable iteration orders.

A security-sensitive change to `src/core/`, `src/format/`, or
`src/tokenizer/` should pass both locally before review.

## Capability negotiation

`eos_caps_t` (returned by `eos_caps()`) lets a host inspect the build:

- which backends, quants, modalities, formats, schedulers, OS shims are
  compiled in
- the SIMD ISAs available at runtime
- the maximum tensor rank and the on-disk format version supported

A model file declares which capabilities it requires; loading a file whose
required capabilities are not in `eos_caps()` fails with
`EOS_E_UNSUPPORTED` rather than silently misbehaving.

## What is **not** in the ABI

- Internal headers under `src/` are not part of the ABI even if
  inadvertently installed.
- The on-disk model format (`.eosm`) is governed by `docs/file_format.md`,
  not by this document.
- Performance: tokens-per-second and latency are not ABI guarantees.
