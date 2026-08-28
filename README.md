# eAI LLM — On-Device LLM Inference Engine

> Repository: `eosllm`. Product name: **eAI LLM**. See [Product identity](#product-identity).

[![CI](https://github.com/embeddedos-org/eosllm/actions/workflows/ci.yml/badge.svg)](https://github.com/embeddedos-org/eosllm/actions/workflows/ci.yml)
[![CodeQL](https://github.com/embeddedos-org/eosllm/actions/workflows/codeql.yml/badge.svg)](https://github.com/embeddedos-org/eosllm/actions/workflows/codeql.yml)
[![Scorecard](https://github.com/embeddedos-org/eosllm/actions/workflows/scorecard.yml/badge.svg)](https://github.com/embeddedos-org/eosllm/actions/workflows/scorecard.yml)
[![Release](https://github.com/embeddedos-org/eosllm/actions/workflows/release.yml/badge.svg)](https://github.com/embeddedos-org/eosllm/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

eosllm is an on-device LLM inference engine written in portable ISO C99. It
compiles to a single static library (`libeosllm.a`) with a stable C ABI
(`include/eosllm/eosllm.h`, `extern "C"` for C++ callers), plus command-line and
server tools that run models entirely locally. It is the **eAI LLM** runtime in
the [EmbeddedOS (EoS)](https://github.com/embeddedos-org) ecosystem's eAI product
family.

> Status per the public header: version 0.1.0, ABI 1. The CLI describes current
> generation as "Phase 1" — greedy text generation from a Llama/Qwen2-class
> GGUF model.

## Product identity

This repository is the implementation of **eAI LLM**, the local language-model
runtime in the eAI product family.

§13.1 of the *EmbeddedOS Platform Architecture & Ecosystem Design Document* v0.9
names the family — eAI Runtime, eAI Vision, eAI Audio, eAI LLM — and says
directly:

> Avoid exposing multiple overlapping names such as eAI, AIL, and eosllm as
> unrelated top-level technologies.

Three names for one capability is a cost paid by everyone evaluating the
platform, so the product name is **eAI LLM** from here on. §31 lists this as a
near-term milestone.

**What does not change.** This is a naming decision, not a technical one:

| | |
|---|---|
| Repository | `embeddedos-org/eosllm` — unchanged |
| Library | `libeosllm.a` — unchanged |
| Public header | `include/eosllm/eosllm.h` — unchanged |
| C ABI | ABI 1 — unchanged |
| Symbol prefix | `eosllm_` — unchanged |
| Tools | `eosllm-cli`, `eosllm-bench` — unchanged |

Renaming the repository, the artifacts or the ABI would break every existing
consumer and is a separate decision that needs its own ADR. Nothing in this
change touches the build.

**Relationship to eAI.** eAI Runtime owns model execution, memory planning and
accelerator abstraction for the general case. eAI LLM is the specialised
language-model runtime alongside it. Neither depends on the other today; §6's
dependency rule applies if that changes — eAI may depend on EoS services, and
EoS must not depend on eAI.


## Features

Observed in the source tree:

- **Model loading** — magic-byte detection with pluggable readers for the native
  `.eosm` format and GGUF (used for bring-up) (`include/eosllm/model.h`,
  `src/format/`).
- **Quantization** — quant kernels and schemes (`src/quant/`,
  `docs/quant_schemes.md`).
- **Inference building blocks** — tokenizer, compute kernels, scheduler, and
  pluggable backends (`src/tokenizer/`, `src/kernels/`, `src/sched/`,
  `include/eosllm/backend.h`).
- **Multimodal hooks** — modality registration (`src/modality/`,
  `include/eosllm/modality.h`).
- **OS abstraction** — a portability layer for bring-up on new targets
  (`src/os/`, `include/eosllm/os.h`, `docs/porting.md`).
- **Editor & browser integrations** — a VS Code extension that shells out to
  `eosllm-cli`, and a browser extension that talks to a local `eosllm-server`
  on `127.0.0.1:7777`.

## What's inside

| Path | Contents |
|------|----------|
| `include/eosllm/` | Public headers (`eosllm.h`, `model.h`, `tensor.h`, `quant.h`, `scheduler.h`, `backend.h`, `modality.h`, `os.h`) |
| `src/` | Engine: `core/`, `format/`, `kernels/`, `quant/`, `sched/`, `tokenizer/`, `modality/`, `os/`, `util/` |
| `tools/` | `eosllm-cli`, `eosllm-server`, `eosllm-bench`, `eosllm-convert`, `eosllm-quant-lab` |
| `tests/` | unit test runner, fuzz targets, and more |
| `browser-extension/` | Chrome/Firefox extension (talks to `eosllm-server`) |
| `vscode-extension/` | VS Code chat extension (wraps `eosllm-cli`) |
| `docs/` | ABI, architecture, file format, CLI, server, porting, quant schemes |

## Build

Requires a C99 compiler (`cc`/`gcc`/`clang`) and `make`.

```bash
make lib        # build libeosllm.a
make tools      # build the CLI, bench, convert, and server binaries
make            # 'all' target: builds the library
make help       # list all targets
```

## Run

```bash
# Greedy generation from a GGUF model
tools/eosllm-cli/eosllm-cli --model <path.gguf> --prompt "Hello" --n 64

# Self-contained smoke test against an in-memory synthetic GGUF (no model file)
tools/eosllm-cli/eosllm-cli --smoke

# Local HTTP/SSE daemon (binds 127.0.0.1:7777 by default)
tools/eosllm-server/eosllm-server
tools/eosllm-server/eosllm-server --port 7878
```

See `docs/cli.md` and `docs/server.md` for the full option and route lists.

## Test

```bash
make test           # build and run the unit test runner
make smoke          # CLI smoke test
make server-smoke   # server liveness smoke test
make bench          # build the benchmark tool
```

## Documentation

Reference docs live under `docs/` (ABI, architecture, file format, porting,
quantization schemes, CLI, and server).

## License

MIT — see [LICENSE](LICENSE).

Part of [embeddedos-org](https://github.com/embeddedos-org).
