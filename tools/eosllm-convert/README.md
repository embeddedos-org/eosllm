# eosllm-convert

Host-only utility that converts HuggingFace `safetensors` checkpoints (or
existing `.gguf` files) into eosllm's native `.eosm` format. Optionally
runs activation-aware calibration (AWQ-style) for sub-2-bit quantization.

Python is permitted here because:

- the eosllm runtime is pure C99 with no third-party deps;
- the converter is host-only and never linked into the engine;
- `safetensors`, `gguf`, and AWQ/GPTQ calibration are not available as
  permissively-licensed C libraries.

## Status

Phase 1 ships a CLI scaffold and `pyproject.toml`. The actual
conversion implementations land in Phase 2 alongside the `.eosm`
format spec (`docs/file_format.md`).

## Subcommands (planned)

- `eosllm-convert from-hf <repo_or_path> --quant q4_k -o model.eosm`
- `eosllm-convert from-gguf <path.gguf> -o model.eosm`     (lossless pass-through)
- `eosllm-convert calibrate --quant q1_58 --dataset wikitext-2 ...`

## Install

```
pip install -e .
```
