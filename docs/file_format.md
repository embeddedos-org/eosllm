# `.eosm` v1 File Format

> **Status (v0.1).** This document is the **byte-level v1 spec**. The
> reader (`src/format/eosm/eosm.c`) currently implements only the
> magic-byte probe; the full reader / writer / converter ship in
> Session 5 of the production roadmap. Once shipped, the implementation
> must round-trip GGUF→`.eosm`→GGUF byte-for-byte (for any built-in
> quant) without re-quantization.

Versioning rule: changing the **format** version (`u32 version` in the
header) is a major break, requires bumping `EOSLLM_ABI_VERSION` (see
`docs/abi.md`), and is documented in `CHANGELOG.md`. Adding new
metadata keys, new dtypes, new capability bits, or new load groups is
**not** a version bump as long as the file remains parseable by a v1
reader (the reader either uses the new bits or surfaces
`EOS_E_UNSUPPORTED`).

Endianness: **little-endian on disk**, always. A reader on a
big-endian host is responsible for byte-swapping every multi-byte
field. There are no architecture-specific flavors of the file.

## 1. Goals (vs. GGUF)

GGUF works for bring-up and the runtime ships a read-only loader for
it. `.eosm` exists to remove four pain points:

1. **Mixed precision per-tensor and per-channel** is first-class. A
   tensor's manifest entry names a `quant_scheme_id` and a
   `calibration_table_idx`; the runtime applies them at read time
   without re-quantization.
2. **Calibration tables are embedded.** AWQ-style per-group scales,
   GPTQ residuals, BitNet ternary masks — all live in their own
   section with a stable layout, not as opaque strings in metadata.
3. **Multi-modal manifest.** One file holds the text decoder, vision
   tower, audio encoder, and fusion adapters; tensors are tagged with
   a `load_group_id` so a session can mmap only the groups it needs.
4. **Stable on-disk hash.** Every file ends with a SHA-256 covering
   everything from the magic up to (but not including) the hash
   itself. A reader rejects any file whose recomputed hash mismatches.

## 2. High-level layout

```
+----------------------+ offset 0
|  header              |  64 bytes, fixed
+----------------------+
|  capability bits     |  variable, declared in header
+----------------------+
|  metadata KV         |  GGUF-compatible key/value pairs
+----------------------+
|  tensor manifest     |  one entry per tensor
+----------------------+
|  calibration tables  |  one entry per referenced calibration
+----------------------+
|  load-group index    |  one entry per load group
+----------------------+
|  load-group A data   |  e.g. text-decoder weights
|  load-group B data   |  e.g. vision-tower weights
|  load-group ...      |
+----------------------+
|  SHA-256 trailer     |  32 bytes; covers [0, file_end - 32)
+----------------------+
```

Section offsets are not in the header. The reader walks the file
linearly: header → capability bits → metadata KV → tensor manifest →
calibration → load-group index → tensor data. Each section starts
immediately after the previous one (no padding) **except** for the
tensor data inside each load group, which is aligned to the
header-declared `alignment`.

## 3. Header (offset 0, fixed 64 bytes)

| Off | Size | Field                        | Notes                                                           |
|----:|-----:|------------------------------|-----------------------------------------------------------------|
|   0 |    4 | `magic`                      | ASCII `"EOSM"` (0x4D, 0x53, 0x4F, 0x45 on disk; little-endian "EOSM"). |
|   4 |    4 | `version`                    | `u32`, current = `1`. Reader rejects `version > MAX_SUPPORTED`. |
|   8 |    8 | `n_capability_bytes`         | `u64`, length of the capability-bits section that follows.     |
|  16 |    8 | `n_metadata_kv`              | `u64`, count of entries in the metadata KV section.            |
|  24 |    8 | `n_tensors`                  | `u64`, count of entries in the tensor manifest.                |
|  32 |    8 | `n_calibration_tables`       | `u64`, count of calibration tables.                            |
|  40 |    8 | `n_load_groups`              | `u64`. Must be ≥ 1. Group `0` is the default group used by every tensor that doesn't specify one. |
|  48 |    4 | `alignment`                  | `u32`. Power of two, ≥ 32, ≤ 65536. Tensor data inside each load group is aligned to this. Default `64`. |
|  52 |    4 | `flags`                      | `u32`. Reserved; must be zero in v1 (a non-zero flags value means the writer used an extension; the reader rejects with `EOS_E_UNSUPPORTED`). |
|  56 |    8 | `_reserved`                  | Must be zero. Reader does not check; future versions may use.  |

### 3.1 Validation

- `magic != "EOSM"` → `EOS_E_FORMAT`.
- `version == 0 || version > 1` → `EOS_E_VERSION_MISMATCH`.
- `alignment` not a power of two, or outside `[32, 65536]` → `EOS_E_FORMAT`.
- `n_load_groups == 0` → `EOS_E_FORMAT`.
- `flags != 0` → `EOS_E_UNSUPPORTED`.

## 4. Capability bits

`n_capability_bytes` raw bytes interpreted as a bitset (LSB of byte 0 is
bit 0). Each set bit declares the file requires a feature; the reader
asks `eos_caps()` whether the linked library has each required bit. If
**any** required bit is not in `eos_caps()`, the reader returns
`EOS_E_UNSUPPORTED` without reading further.

| Bit | Symbol                        | Meaning                                                        |
|----:|-------------------------------|----------------------------------------------------------------|
|   0 | `CAP_QUANT_Q8_0`              | Some tensor uses `q8_0`.                                       |
|   1 | `CAP_QUANT_Q4_K`              | Some tensor uses `q4_k`.                                       |
|   2 | `CAP_QUANT_Q2_K`              | Some tensor uses `q2_k`.                                       |
|   3 | `CAP_QUANT_Q1_58`             | Some tensor uses BitNet 1.58 ternary.                          |
|   4 | `CAP_QUANT_MIXED`             | At least one tensor uses a per-channel mixed scheme.           |
|   5 | `CAP_QUANT_CALIBRATED`        | At least one tensor references a calibration table.            |
|   8 | `CAP_MODALITY_TEXT`           | File contains a text decoder.                                  |
|   9 | `CAP_MODALITY_VISION`         | File contains a vision tower.                                  |
|  10 | `CAP_MODALITY_AUDIO`          | File contains an audio encoder.                                |
|  16 | `CAP_FUSION_ADAPTERS`         | File contains modality-fusion adapter weights.                 |
|  24 | `CAP_LOAD_GROUPS_GT_1`        | File has more than one load group; reader must support partial loads to use it. |
|  32 | `CAP_HASH_SHA256_TRAILER`     | The 32-byte trailer is a SHA-256 over `[0, file_end - 32)`. Required in v1; explicit so future formats can drop it. |

Bits 33..63 are reserved for v1; bit 63 is the **experimental** bit
(must be zero in any released file). Bits beyond byte 7 are reserved
for v2+ and ignored by a v1 reader unless `flags != 0`.

## 5. Metadata KV section

Layout per entry (immediately following the capability section):

```
struct kv {
    u64        key_len;           // bytes
    u8         key[key_len];      // UTF-8, no null terminator
    u32        value_type;        // see KV_TYPE table
    union {                       // discriminated by value_type
        u8/u16/u32/u64 scalar;
        i8/i16/i32/i64 scalar;
        f32/f64        scalar;
        u8             boolean;
        struct { u64 len; u8 bytes[len]; }                          string;
        struct { u32 elem_type; u64 count; <count elements>; }      array;
    } value;
};
```

`KV_TYPE` enum is a superset of GGUF's:

| Value | Type     | Notes                                                           |
|------:|----------|-----------------------------------------------------------------|
|     0 | `U8`     | 1 byte                                                          |
|     1 | `I8`     | 1 byte                                                          |
|     2 | `U16`    | 2 bytes                                                         |
|     3 | `I16`    | 2 bytes                                                         |
|     4 | `U32`    | 4 bytes                                                         |
|     5 | `I32`    | 4 bytes                                                         |
|     6 | `F32`    | 4 bytes                                                         |
|     7 | `BOOL`   | 1 byte (any non-zero value means true)                          |
|     8 | `STRING` | `u64 len` + `len` bytes UTF-8                                   |
|     9 | `ARRAY`  | `u32 elem_type` + `u64 count` + element data                    |
|    10 | `U64`    | 8 bytes                                                         |
|    11 | `I64`    | 8 bytes                                                         |
|    12 | `F64`    | 8 bytes                                                         |
|    13 | `BLOB`   | `u64 len` + `len` opaque bytes (NEW in `.eosm`; not in GGUF)    |

`BLOB` exists for keys like `tokenizer.regex.pre_compiled` where we
want to ship a precompiled DFA without a UTF-8 sanity check forced on
it.

### 5.1 Required keys (v1)

| Key                           | Type     | Constraint                                                 |
|-------------------------------|----------|------------------------------------------------------------|
| `general.architecture`        | STRING   | One of `"llama"`, `"qwen2"`, `"phi"`, `"bitnet"`, `"llava"`, `"whisper"`. |
| `general.eosm.version`        | U32      | Must equal the header's `version`.                         |
| `general.alignment`           | U32      | Must equal the header's `alignment`.                       |
| `tokenizer.ggml.model`        | STRING   | `"bpe"` or `"spm"`. Required if `CAP_MODALITY_TEXT` set.   |

Optional GGUF-compatible keys (`tokenizer.ggml.*`, `llama.context_length`,
etc.) carry the same meaning as in GGUF.

## 6. Tensor manifest

One entry per tensor, in the order tensors appear in load groups
(grouped by `load_group_id`, then sorted by `offset` within each
group). Layout:

```
struct tensor_manifest_entry {
    u64    name_len;                  // bytes
    u8     name[name_len];            // UTF-8
    u32    dtype;                     // EOS_DT_* enum value
    u32    quant_scheme_id;           // u32 hash of the scheme name; 0 = none (raw f32/f16)
    u32    calibration_table_idx;     // u32; 0xFFFFFFFF = none
    u8     rank;                      // 1..EOS_MAX_RANK
    u8     load_group_id;             // 0..n_load_groups-1
    u8     _reserved[2];              // must be zero
    u64    dims[rank];                // u64 each; padded if rank < EOS_MAX_RANK is NOT done — reader walks exactly rank
    u64    offset;                    // bytes from the start of the load group's data region
    u64    data_bytes;                // exact byte count for this tensor
};
```

`quant_scheme_id` is the FNV-1a 32-bit hash of the scheme's name
string (e.g. `"q4_k"`). The reader looks the id up in a small table at
startup; unknown ids → `EOS_E_UNSUPPORTED`.

`data_bytes` is **stored** rather than recomputed because mixed quants
make recomputation depend on the calibration table contents. The
reader still validates `data_bytes` against the scheme's expected
block size and rejects mismatches.

## 7. Calibration tables

One entry per table, in order. Tables are deduplicated: if 100 tensors
share the same per-group AWQ scale, they all reference one table.

```
struct calibration_table {
    u32   table_type;          // see CALIB_TYPE enum
    u32   _reserved;           // must be zero
    u64   data_bytes;
    u8    data[data_bytes];    // table-type-specific payload
};
```

| Value | `CALIB_TYPE` | Payload                                                                  |
|------:|--------------|--------------------------------------------------------------------------|
|     0 | `NONE`       | Sentinel; table not used. Writers should not emit; readers tolerate.     |
|     1 | `AWQ_PER_GROUP_SCALES_F16` | `n_groups` × `f16` per-group scales. Group size declared per tensor in metadata. |
|     2 | `GPTQ_RESIDUAL_F16`        | `n_elements` × `f16` residuals.                                         |
|     3 | `BITNET_TERNARY_MASK`      | Packed 2-bits-per-value mask: `(n_elements + 3) / 4` bytes.              |

Adding a new `CALIB_TYPE` is additive (no version bump); a reader that
doesn't recognize a type returns `EOS_E_UNSUPPORTED` for any tensor
that references it.

## 8. Load-group index

A small table that lets the reader mmap a subset of the file:

```
struct load_group_index_entry {
    u8     id;                  // matches load_group_id in tensor manifest
    u8     _reserved[7];
    u64    data_offset;         // bytes from start of file
    u64    data_bytes;          // bytes
    u64    name_len;
    u8     name[name_len];      // UTF-8 label, e.g. "text", "vision", "audio"
};
```

The reader iterates the index in declaration order; tensors are
resolved by adding their `offset` to the matching group's
`data_offset`.

## 9. Tensor data

Each load group's data block is byte-aligned to the header's
`alignment`. Tensors within a group are packed contiguously in
manifest order, also each aligned to `alignment`. There is **no**
per-tensor padding bookkeeping in the manifest; `offset` in the
manifest is the actual on-disk offset within the group's data block,
already accounting for alignment.

## 10. SHA-256 trailer

The last 32 bytes of the file are a SHA-256 over the file `[0, len -
32)`. The reader recomputes the hash on open; mismatch →
`EOS_E_FORMAT`. The hash is not optional in v1 — `CAP_HASH_SHA256_TRAILER`
must be set.

For very large files (> 16 GiB) where streaming the whole file through
SHA-256 at open is too costly, a future revision may add a
`CAP_HASH_INCREMENTAL` bit and a per-load-group hash; v1 readers will
reject any file with that bit set.

## 11. Compatibility & coexistence with GGUF

Phase 1 ships a read-only GGUF v3 reader (`src/format/gguf/`) so the
runtime can bring up models against the existing ecosystem. `.eosm`
and GGUF coexist via the format-registry vtable; the runtime probes
each format on `eos_model_open`. The `.eosm` writer's lossless
GGUF→`.eosm` path (Session 5 / E4) must satisfy:

```
eos_model_open(gguf_path) -> logits_a
eosllm-convert from-gguf x.gguf -o /tmp/x.eosm
eos_model_open(/tmp/x.eosm) -> logits_b
assert logits_a == logits_b   (bit-exact for f32 / f16 weights;
                               within 1e-4 max-abs-diff for q4_k)
```

This is the acceptance test that Session 5 must pass before the v1
spec is considered settled.

## 12. Why these specific design choices

- **Header is exactly 64 bytes** — fits in one cache line on every
  current arch and lets the reader read it in one `pread`.
- **No section offsets in the header** — section sizes change as the
  metadata + tensor manifest change; the reader has to walk anyway,
  and storing offsets duplicates information already in `n_*` counts.
- **SHA-256, not CRC** — files are model weights downloaded from
  third-party hosts; we want collision resistance, not just
  bit-error detection.
- **`quant_scheme_id` is an FNV-1a hash, not an enum** — third-party
  quant schemes can register their own scheme name without coordinating
  with the eosllm enum.
- **Calibration tables are deduplicated** — for AWQ-quantized 8B
  models, hundreds of tensors share the same per-group scale tensor;
  storing it once saves >100 MB.
- **Load groups, not module sections** — the unit of "loadable as a
  unit" is finer-grained than the unit of "modality": a Llava model
  has text-decoder, vision-tower, mm-projector, all of which are
  separately mappable.

## 13. Open questions for v2

- Per-load-group SHA-256 to support partial loads without rehashing
  the whole file.
- Compressed metadata KV section (the vocab can be hundreds of KB of
  UTF-8 strings).
- A "tile index" for very large tensors so a session can lazily fault
  in only the rows it needs (relevant for paged-KV /
  expert-of-mixture).
- Signed manifest (Ed25519 over the header + manifest, separate from
  the SHA-256 trailer) for distribution-channel signing.

These are deliberately deferred. Adding any of them to v1 increases
the implementation cost without serving the v0.1 release goals.
