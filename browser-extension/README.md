# eosllm — browser extension

Local-first LLM chat in Chrome / Firefox. The extension talks to a
local `eosllm-server` daemon on `127.0.0.1:7777` (loopback only by
default); nothing is transmitted off your machine.

## Install (end users)

### Chrome / Edge / other Chromium-based

1. Install from the [Chrome Web Store](https://chromewebstore.google.com/detail/eosllm/PLACEHOLDER)
   (PLACEHOLDER — link populated when first published).
2. Pin the eosllm icon to the toolbar.
3. Click the icon → click *settings* → confirm `Server URL` is
   `http://127.0.0.1:7777` and (optionally) set `Model path` to a
   GGUF file readable by the server.

### Firefox

1. Install from [Mozilla Add-ons](https://addons.mozilla.org/firefox/addon/eosllm/) (PLACEHOLDER).

### Sideload (any Chromium)

If you'd rather sideload from a release `.zip`:

1. Download `eosllm-chrome-X.Y.Z.zip` from the [GitHub Releases page](https://github.com/embeddedos-org/eosllm/releases).
2. Extract.
3. `chrome://extensions` → toggle *Developer mode* → *Load unpacked* →
   point at the extracted folder.

## Run the server

The extension assumes you have `eosllm-server` running locally. From
the engine release tarball:

```
./eosllm-server                                    # default 127.0.0.1:7777
./eosllm-server --allow-origin chrome-extension://YOUR_EXT_ID
```

The `--allow-origin` argument is needed for Chrome — it's the only way
the daemon's CORS layer will accept fetches from your extension.

To find your extension's ID, open `chrome://extensions` and look at
the eosllm card.

## Develop

```
cd browser-extension
npm install
npm run build
# unpacked extension is now under dist/; load it via
# chrome://extensions → Load unpacked → point at dist/
```

To package for the stores:

```
npm run package-chrome     # → eosllm-chrome-X.Y.Z.zip
npm run package-firefox    # → eosllm-firefox-X.Y.Z.xpi
```

## Architecture

```
+----------------+        chrome.runtime           +----------------+
| popup.ts       |<-----  message channel  ------> | background.ts  |
| popup.html     |                                 | (service       |
+----------------+                                 |  worker)       |
        ^                                          +----------------+
        |  postMessage('token','done')                       |
                                                              |  fetch
                                                              v
                                                  +-------------------+
                                                  |  eosllm-server    |
                                                  |  on localhost     |
                                                  +-------------------+
```

The background service worker holds the SSE stream so the popup can be
closed and reopened mid-generation without losing tokens.

## Privacy

- Network: extension only ever fetches `${serverUrl}/generate`. Default
  is `http://127.0.0.1:7777`. No telemetry. No analytics. No phone
  home.
- Storage: `chrome.storage.local` only; nothing in `sync` storage.
- Permissions: `["storage"]` and `["http://localhost:7777/*"]` only.
  No `tabs`, no `webRequest`, no content-script injection.

## License

[MIT](../vscode-extension/LICENSE) (same as the engine).
