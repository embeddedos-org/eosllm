# eosllm — VS Code extension

Local-first LLM chat in VS Code, powered by the
[eosllm](https://github.com/embeddedos-org/eosllm) inference engine.

The extension wraps the `eosllm-cli` binary in shell-out mode and
streams tokens into a chat webview as they're generated. Everything
runs locally; no network round-trip.

## Quickstart

1. Install the extension from the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=eosllm.eosllm-vscode)
   or the [Open VSX Registry](https://open-vsx.org/extension/eosllm/eosllm-vscode).
2. On first launch the extension downloads the platform-matching
   `eosllm-cli` binary from the corresponding GitHub Release tag,
   placing it under the extension's global storage.
3. Open VS Code Settings (`Ctrl/Cmd+,`) → search **eosllm** → set
   `eosllm.modelPath` to a Llama-class GGUF on disk.
4. Open the eosllm activity-bar view, type a prompt, hit **Send**.

## Settings

| Setting | Default | Notes |
|---|---|---|
| `eosllm.modelPath` | (empty) | Absolute path to the GGUF. Required. |
| `eosllm.maxTokens` | `256` | Per-generation upper bound. |
| `eosllm.binDir` | (empty) | Override the bundled binary location. Power users only. |

## Commands

| Command | Default keybinding | Notes |
|---|---|---|
| `eosllm: Run prompt` | (none) | Quick-pick prompt entry; result shown in chat panel. |
| `eosllm: Open chat panel` | (none) | Focus the activity-bar chat view. |

## Architecture

```
+------------------+        spawn         +-----------------+
| extension.ts     |--------------------->| eosllm-cli      |
|  EosllmSession   |   --stream-jsonl     |   (native)      |
|                  |<---------------------|                 |
+------------------+   one JSON / line    +-----------------+
        |
        |  onToken events
        v
+------------------+
| chat webview     |
+------------------+
```

The wire-protocol contract (one JSONL line per token, terminator with
`done:true`) is documented in
[docs/cli.md](https://github.com/embeddedos-org/eosllm/blob/main/docs/cli.md)
and frozen as part of the engine's v1.0.0 ABI.

## Privacy

Nothing is transmitted off your machine. The extension does not phone
home, does not send telemetry, does not contact any server other than
GitHub Releases (only on first launch, to fetch the engine binary
matching your platform).

## Development

```
cd vscode-extension
npm install
npm run compile
code --extensionDevelopmentPath=.   # F5 from inside VS Code
```

## License

[MIT](LICENSE).
