# Benchmarks

> **Status (v0.1).** A self-serve matmul microbench is implemented and
> wired into CI. End-to-end LLM benchmarks (tokens/sec on real models)
> are tracked separately and require user-supplied model files.

## What's actually measured today

`tools/eosllm-bench` runs `matmul_f32`, `matmul_q8_0`, and `matmul_q4_k`
across every registered backend whose runtime probe passes, at six
fixed shapes (M = 1, N ∈ {1, 8, 64, 128, 256, 512}, K ∈ {256 … 4096}).
Per-shape iteration count is auto-tuned to ~50 ms of work. Output is
JSON: `{ library_version, abi_version, backends_probed, records[] }`
where each record is `(backend, op, m, n, k, n_iters, total_ns,
ns_per_iter, giga_ops_per_s)`.

## Running it

```
make benchmark
```

Auto-detects host ISA (`uname -m`):

| arch         | flag added                       |
|--------------|----------------------------------|
| `x86_64`     | `EOSLLM_HAVE_KERNEL_AVX2=1`      |
| `aarch64`    | `EOSLLM_HAVE_KERNEL_NEON=1`      |
| other        | scalar only                      |

Output lands at `docs/benchmarks/host.json` (gitignored — volatile per host).

`make benchmark` does a `make clean` first so the autoconf flags
actually take effect; expect a 10-30 s wall time.

## Comparing against a baseline

```
# Once: snapshot today's numbers as the reference for this CPU.
cp docs/benchmarks/host.json docs/benchmarks/baseline-x86_64.json

# Later: run the bench again and diff.
make bench-diff
```

`make bench-diff` runs `make benchmark` and then
`tools/bench_diff.py docs/benchmarks/baseline-$(HOST_ARCH).json
docs/benchmarks/host.json`. Override the baseline path with
`BASELINE=...`. Exits non-zero if any record's
`giga_ops_per_s` regresses by more than the threshold (default 10 %).

The default `.gitignore` lets baselines live in-repo via the
`docs/benchmarks/baseline-*.json` whitelist; run-time `host.json`
files do not.

## Sanitizer-clean bench

```
make sanitize-bench
```

Builds the bench with `-fsanitize=address,undefined`,
`ASAN_OPTIONS=detect_leaks=1`, and runs the full microbench plus
`--shapes-only`. Catches kernel-level bugs that the unit suite
misses (unaligned loads in AVX2/NEON intrinsics, OOB reads on the
synthetic Q4_K weight buffer, leaks across the bench's own
malloc cycle). Bar: zero ASan/UBSan reports.

## CI

The `benchmark` job in `.github/workflows/ci.yml` runs `make
benchmark` on every PR, `cat`s the JSON to the job log, and uploads
`docs/benchmarks/host.json` as an artifact named
`bench-${{ github.sha }}` via `actions/upload-artifact@v4`. Reviewers
can download two artifacts and run `tools/bench_diff.py` locally to
see per-shape deltas across PRs.

## Reference numbers (informational, will vary by CPU)

These are from a single run on the developer machine
(AMD Threadripper PRO 3975WX, Linux x86_64, gcc 9, debug build).
**Not committed as a baseline** — your CPU will produce different
absolute numbers; what matters is intra-host stability and the AVX2
vs scalar ratios.

| op           | shape (M×N×K) | scalar GOPS | x86_avx2 GOPS | speedup |
|--------------|---------------|-------------|---------------|--------:|
| matmul_f32   | 1×1×256       | 0.70        | 2.20          | 3.1×    |
| matmul_f32   | 1×128×2048    | 0.71        | 2.58          | 3.6×    |
| matmul_f32   | 1×256×4096    | 0.71        | 2.62          | 3.7×    |
| matmul_q8_0  | 1×128×2048    | 0.70        | (scalar only) | —       |
| matmul_q4_k  | 1×128×2048    | 0.68        | 0.97          | 1.4×    |

Notes on the modest Q4_K speedup: at M=1 and these K values the
kernel is **memory-bandwidth-bound** on the Q4_K weight buffer, not
compute-bound. The AVX2 path saves wall time on the f32 math but
not on byte traffic. The future int8 fused path
(`_mm256_maddubs_epi16` over packed quants) is expected to roughly
double Q4_K throughput once it lands.

## End-to-end LLM benchmarks (deferred)

Tokens/sec, time-to-first-token, peak RSS on real models — the
"benchmarks ≥ 3 model × HW combinations" target from
`~/.llms/plans/eosllm_production_roadmap.plan.md` Session 14 — is
deferred until model files are dropped into `tests/data/`. The
current bench is the regression net for the kernels themselves;
end-to-end will reuse the same JSON shape with new record types.
