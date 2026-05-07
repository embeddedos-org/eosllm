/*
 * browser-extension/src/popup.ts
 *
 * Popup chat UI. Sends prompts to background.ts via
 * chrome.runtime.sendMessage; listens for "token" / "done" replies
 * and renders them inline.
 */

const log = document.getElementById("log")!;
const inp = document.getElementById("prompt") as HTMLInputElement;
const btn = document.getElementById("send")!;
const opt = document.getElementById("open-options")!;

let assistantSpan: HTMLSpanElement | null = null;

function append(node: Node): void {
    log.appendChild(node);
    log.scrollTop = log.scrollHeight;
}

function send(): void {
    const text = inp.value.trim();
    if (text.length === 0) { return; }
    inp.value = "";

    const u = document.createElement("div");
    u.className = "user";
    u.textContent = "› " + text;
    append(u);

    assistantSpan = document.createElement("span");
    const wrap = document.createElement("div");
    wrap.appendChild(assistantSpan);
    append(wrap);

    void chrome.runtime.sendMessage({
        kind: "generate", prompt: text, maxTokens: 256,
    });
}

btn.addEventListener("click", send);
inp.addEventListener("keydown", (e: KeyboardEvent) => {
    if (e.key === "Enter") { send(); }
});

opt.addEventListener("click", (e: MouseEvent) => {
    e.preventDefault();
    void chrome.runtime.openOptionsPage();
});

chrome.runtime.onMessage.addListener((msg: {
    kind: string; t?: string; reason?: string;
    n_tokens?: number; detail?: string;
}) => {
    if (msg.kind === "token" && assistantSpan !== null) {
        assistantSpan.appendChild(document.createTextNode(msg.t ?? ""));
        log.scrollTop = log.scrollHeight;
    } else if (msg.kind === "done") {
        const m = document.createElement("div");
        m.className = "meta";
        m.textContent = "[done: " + (msg.reason ?? "?") + ", "
            + (msg.n_tokens ?? 0) + " tokens"
            + (msg.detail !== undefined ? " — " + msg.detail : "")
            + "]";
        append(m);
        assistantSpan = null;
    }
});
