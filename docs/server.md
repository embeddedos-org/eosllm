# eosllm-server

A single-binary local HTTP/SSE daemon that exposes the eosllm engine
to clients that can't (or don't want to) embed the C library directly
— most importantly the browser extension under `browser-extension/`,
which talks to it on `localhost:7777`.

## Quickstart

```
./eosllm-server                          # bind 127.0.0.1:7777
./eosllm-server --port 7878              # custom port
./eosllm-server --allow-origin chrome-extension://abcd...
```

## Routes

### `GET /healthz`

Liveness probe. Always returns `200 OK` with body `ok\n`. Used by
`make server-smoke` and by clients to detect the daemon is up before
opening an SSE stream.

### `GET /caps`

Capability inspection. Mirrors the JSON emitted by
`eosllm-cli --caps` (see [`docs/cli.md`](cli.md)) plus a
`server_version` field. Use this from a client to feature-gate which
backends to request.

```
$ curl http://127.0.0.1:7777/caps | jq .library_version
"0.1.0"
```

### `POST /generate`

Token-streaming generation. Request body is a JSON object:

```
{
  "prompt": "Hello, ",
  "max_tokens": 64,
  "model": "/path/to/model.gguf"   // optional; see below
}
```

| Key | Required | Default | Notes |
|---|---|---|---|
| `prompt` | yes | — | UTF-8 string. Tokenized via the model's tokenizer. |
| `max_tokens` | no | 64 | Upper bound on emitted tokens. Capped at 8192. |
| `model` | no | — | Path on the daemon's filesystem to the model file. When omitted the daemon runs the smoke flow (NULL model — useful for protocol smoke tests without a real `.gguf` at hand). |

Response is `text/event-stream` (SSE). Each token is delivered as:

```
event: token
data: {"t":"foo","i":42}
```

After the last token (or on error) a single `done` event is emitted:

```
event: done
data: {"reason":"max","n_tokens":64}
```

`reason` is one of:

- `"max"` — the requested `max_tokens` was reached.
- `"eos"` — the model emitted an end-of-stream token.
- `"error"` — generation failed; see `detail` field for the engine's
  `eos_status_str()` value.

## Security model

By default the daemon binds **127.0.0.1 only**. To expose it to the
local network, pass `--bind 0.0.0.0` — but then add `--allow-origin`
discipline and consider running behind a TLS reverse proxy.

CORS:

- Default is **deny** (no `Access-Control-Allow-Origin` header).
- Pass `--allow-origin <origin>` to whitelist exactly one origin
  (e.g. `chrome-extension://aiabcdefghij...`).
- The browser extension's installer prints the right argument
  for you; see `browser-extension/README.md`.

Other limits:

- Request body capped at **64 KiB** (returns `413` otherwise).
- Single-threaded, one connection at a time. Sufficient for a local
  developer-loopback workload (the engine is single-threaded by
  default anyway).
- No authentication. The loopback bind is the only access control;
  treat any non-loopback exposure as untrusted.

## Operational notes

The daemon runs in the foreground; backgrounding is the operator's
responsibility (`./eosllm-server &`, `systemd --user`, etc.).

Logging goes to stderr (level-prefixed via the engine's POSIX log
shim). The HTTP/SSE protocol itself logs a single startup line to
stdout containing the bound URL.

## Wire-protocol contract

This file is the stable contract for the wire protocol. Every change
to:

- the route set,
- the request body schema,
- the SSE `event` names,
- the JSON keys inside event `data`,

is an ABI-style change that bumps `EOSLLM_SERVER_VERSION` in
`tools/eosllm-server/main.c` AND adds a `[Changed]` entry under
the corresponding release in `CHANGELOG.md`.
