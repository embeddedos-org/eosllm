/*
 * browser-extension/src/background.ts
 *
 * MV3 service worker. Acts as the EventSource holder so the popup can
 * close without losing the in-flight stream.
 *
 * Wire protocol: SSE from POST /generate (see docs/server.md in the
 * engine repo). Each `event: token` carries
 * `{"t":"<text>","i":<token_id>}`; the final `event: done` carries
 * `{"reason":"...","n_tokens":N}`.
 *
 * Note: Chrome MV3 service workers can be killed at any time. We
 * intentionally don't keep state across worker restarts; the popup
 * must re-issue its prompt if the worker was suspended mid-stream.
 */

interface GenerateRequest {
    kind: "generate";
    prompt: string;
    maxTokens: number;
    serverUrl?: string;
    modelPath?: string;
}

interface CancelRequest {
    kind: "cancel";
}

interface PingRequest {
    kind: "ping";
}

type IncomingMessage = GenerateRequest | CancelRequest | PingRequest;

interface Settings {
    serverUrl: string;   // default "http://127.0.0.1:7777"
    modelPath: string;   // optional; if empty, server runs synthetic smoke
    maxTokens: number;
}

const DEFAULT_SETTINGS: Settings = {
    serverUrl: "http://127.0.0.1:7777",
    modelPath: "",
    maxTokens: 256,
};

async function loadSettings(): Promise<Settings> {
    const got = await chrome.storage.local.get(DEFAULT_SETTINGS);
    return { ...DEFAULT_SETTINGS, ...got } as Settings;
}

// ---------------------------------------------------------------------
// Active stream bookkeeping
// ---------------------------------------------------------------------

let active: AbortController | null = null;

function broadcastToken(text: string, id: number): void {
    void chrome.runtime.sendMessage({
        kind: "token", t: text, i: id,
    }).catch(() => { /* popup may be closed */ });
}

function broadcastDone(reason: string, nTokens: number, detail?: string): void {
    void chrome.runtime.sendMessage({
        kind: "done", reason, n_tokens: nTokens, detail,
    }).catch(() => { /* popup may be closed */ });
}

// ---------------------------------------------------------------------
// SSE streaming via fetch + ReadableStream
// MV3 service workers can't use EventSource (no global SSE constructor
// in the worker scope as of Chrome 120), so we parse SSE manually.
// ---------------------------------------------------------------------

async function streamGenerate(
    settings: Settings,
    prompt: string,
    maxTokens: number,
): Promise<void> {
    if (active !== null) {
        active.abort();
        active = null;
    }
    const ctl = new AbortController();
    active = ctl;

    const url  = `${settings.serverUrl.replace(/\/$/, "")}/generate`;
    const body: Record<string, unknown> = {
        prompt, max_tokens: maxTokens,
    };
    if (settings.modelPath.length > 0) { body.model = settings.modelPath; }

    let nTokens = 0;

    let res: Response;
    try {
        res = await fetch(url, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify(body),
            signal:  ctl.signal,
        });
    } catch (err: unknown) {
        broadcastDone("error", 0,
            `fetch failed: ${(err as Error).message}`);
        return;
    }

    if (!res.ok || !res.body) {
        broadcastDone("error", 0, `http ${res.status}`);
        return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buf = "";
    let currentEvent = "message";

    try {
        for (;;) {
            const { value, done } = await reader.read();
            if (done) { break; }
            buf += decoder.decode(value, { stream: true });

            // Process complete SSE events (terminated by \n\n).
            let sep: number;
            // eslint-disable-next-line no-cond-assign
            while ((sep = buf.indexOf("\n\n")) >= 0) {
                const block = buf.slice(0, sep);
                buf = buf.slice(sep + 2);
                let dataLine = "";
                for (const ln of block.split("\n")) {
                    if (ln.startsWith("event:")) {
                        currentEvent = ln.slice(6).trim();
                    } else if (ln.startsWith("data:")) {
                        dataLine = ln.slice(5).trim();
                    }
                }
                if (dataLine.length === 0) { continue; }
                try {
                    const obj = JSON.parse(dataLine);
                    if (currentEvent === "token" && typeof obj.t === "string") {
                        nTokens++;
                        broadcastToken(obj.t, obj.i ?? 0);
                    } else if (currentEvent === "done") {
                        broadcastDone(obj.reason ?? "max",
                                       obj.n_tokens ?? nTokens,
                                       obj.detail);
                        return;
                    }
                } catch (_e) {
                    /* ignore malformed event */
                }
            }
        }
        // Stream ended without a `done` event — synthesize one.
        broadcastDone("max", nTokens);
    } catch (err: unknown) {
        if ((err as Error).name === "AbortError") {
            broadcastDone("user", nTokens);
        } else {
            broadcastDone("error", nTokens, (err as Error).message);
        }
    } finally {
        if (active === ctl) { active = null; }
    }
}

// ---------------------------------------------------------------------
// Message router
// ---------------------------------------------------------------------

chrome.runtime.onMessage.addListener((
    msg: IncomingMessage,
    _sender: chrome.runtime.MessageSender,
    sendResponse: (resp: unknown) => void,
): boolean => {
    if (msg.kind === "ping") {
        sendResponse({ ok: true });
        return false;
    }
    if (msg.kind === "cancel") {
        if (active !== null) { active.abort(); active = null; }
        sendResponse({ ok: true });
        return false;
    }
    if (msg.kind === "generate") {
        loadSettings().then((s) =>
            streamGenerate(s, msg.prompt, msg.maxTokens)
        ).catch((e: Error) =>
            broadcastDone("error", 0, e.message));
        sendResponse({ ok: true });
        return true;   // keep the channel open for follow-up messages
    }
    return false;
});

// Optional: listen for explicit shutdown signal.
chrome.runtime.onSuspend.addListener(() => {
    if (active !== null) { active.abort(); active = null; }
});

// Self-test that the worker loaded; no UI side effect.
console.log("[eosllm] background service worker ready");
