# eosllm-cli

A minimal text-generation CLI for the eosllm engine. Bundled in every
release tarball under `bin/eosllm-cli`.

## Quickstart

```
eosllm-cli --model /path/to/model.gguf --prompt "Hello, " --max-tokens 64
```

## Modes

### Generation

```
eosllm-cli --model <path.gguf> --prompt "<text>" \
           [--max-tokens N] [--n N]              \
           [--ctx N] [--scheduler greedy]        \
           [--stream-jsonl]
```

| Flag | Default | Notes |
|---|---|---|
| `--model` | (required) | Path to a Llama-class GGUF v3 file. |
| `--prompt` | (required) | UTF-8 prompt; tokenized via the model's tokenizer. |
| `--max-tokens` / `--n` | `64` | Upper bound on emitted tokens. Both names accepted. |
| `--ctx` | `2048` | `eos_session_params_t::max_context`. |
| `--scheduler` | `greedy` | Currently only `greedy` is wired up. |
| `--stream-jsonl` | off | Emit one JSON-Lines record per token to stdout (see schema below). |

### Smoke / diagnostics (no model required)

| Flag | Purpose |
|---|---|
| `--smoke` | In-memory synthetic-GGUF lifecycle. |
| `--smoke --stream-jsonl` | Emit a 3-token synthetic JSONL stream — useful as a wire-protocol smoke for clients (VS Code extension, browser extension). |
| `--smoke-bad-magic` | Negative path: GGUF reader rejects "XXXX" magic. |
| `--smoke-empty-stream` | Negative path: 0-byte stream rejected at first read. |
| `--last-error` | Calls `eos_model_open` on a non-existent path; prints both `eos_status_str` and `eos_last_error`. |
| `--caps` | Compile-time + runtime capability JSON. |
| `--metadata <path>` | Dump every well-known metadata key from a GGUF as JSON. |
| `--version` / `-V` | `eosllm <version> abi=<n>`. |

## JSONL streaming protocol

`--stream-jsonl` is the **stable contract** consumed by every editor /
browser surface that wraps the CLI. Each generation produces a sequence
of single-line JSON records on stdout, terminated by a single `done`
record:

### Per-token record

```json
{"t":"<utf8-text>","i":<token_id>}
```

| Key | Type | Notes |
|---|---|---|
| `t` | string | Tokenizer-decoded UTF-8 text for this token. JSON-escaped: `"` → `\"`, `\` → `\\`, `\n` → `\\n`, `\r` → `\\r`, `\t` → `\\t`. Other control bytes are dropped. |
| `i` | integer | Raw token id from the model's vocabulary. |

### Terminator record

```json
{"done":true,"reason":"<reason>","n_tokens":<n>,"ms":<m>,"tok_per_s":<t>}
```

| Key | Type | Notes |
|---|---|---|
| `done` | bool | Always `true`. Disambiguates from per-token records. |
| `reason` | string | `"eos"` (model emitted EOS), `"max"` (`--max-tokens` reached), `"user"` (deadline elapsed), `"error"` (engine failure). |
| `n_tokens` | int | Tokens actually emitted before the terminator. |
| `ms` | int | Wall time in milliseconds. Currently always `0` (TODO: wire `eosi_now_ns`). |
| `tok_per_s` | int | Throughput in tokens per second. Currently always `0`. |

On `reason: "error"` the record may carry an additional `detail` string
naming the failing engine call (`feed`, `step`, `model_open`).

### Stability

The schema is **stable per [docs/abi.md](abi.md)**:

- New optional keys may be appended in minor releases. Consumers MUST
  ignore unknown keys.
- The `t` and `i` keys never change shape.
- The presence of exactly one terminator record per stream is invariant.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success (or expected-failing diagnostic mode succeeded). |
| `1` | Engine failure (model open / session open / generation). |
| `2` | Bad arguments / usage error. |

## See also

- [docs/abi.md](abi.md) — public C ABI surface.
- [docs/server.md](server.md) — `eosllm-server` HTTP/SSE wrapper that
  reuses the same engine.
- [docs/release.md](release.md) — release pipeline + signing.
