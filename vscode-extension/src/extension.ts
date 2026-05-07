/*
 * vscode-extension/src/extension.ts
 *
 * Entry point. Wires:
 *   - the `eosllm.runPrompt` command (quick-pick prompt → status output)
 *   - the `eosllm.openChat` command (focuses the chat webview)
 *   - the `eosllm.chatView` webview view (streamed chat UI)
 *   - EosllmSession: spawns `eosllm-cli --stream-jsonl` per prompt and
 *     emits onToken / onDone events as JSONL lines arrive on stdout
 *   - status-bar token-rate readout
 */

import * as vscode from "vscode";
import * as path from "path";
import * as fs from "fs";
import * as os from "os";
import { spawn, ChildProcess } from "child_process";
import { EventEmitter } from "events";

// ---------------------------------------------------------------------
// JSONL stream wrapper
// ---------------------------------------------------------------------

interface TokenLine {
    t: string;
    i: number;
}
interface DoneLine {
    done: true;
    reason: "eos" | "max" | "user" | "error";
    n_tokens: number;
    ms: number;
    tok_per_s: number;
    detail?: string;
}

/**
 * Spawns `eosllm-cli --stream-jsonl ...` and parses its line-delimited
 * JSON output. Emits "token" per token and "done" exactly once.
 *
 * Stable contract: see docs/cli.md in the engine repo.
 */
class EosllmSession extends EventEmitter {
    private proc: ChildProcess | null = null;
    private buf = "";

    constructor(
        private readonly cliPath: string,
        private readonly modelPath: string,
        private readonly maxTokens: number,
    ) {
        super();
    }

    start(prompt: string): void {
        const args = [
            "--model",       this.modelPath,
            "--prompt",      prompt,
            "--max-tokens",  String(this.maxTokens),
            "--stream-jsonl",
        ];
        this.proc = spawn(this.cliPath, args, {
            stdio: ["ignore", "pipe", "pipe"],
        });

        // stdio above guarantees stdout + stderr are piped, but
        // ChildProcess's static type doesn't narrow that. Guard
        // explicitly so strict-mode tsc is happy.
        if (this.proc.stdout === null || this.proc.stderr === null) {
            this.emit("done", {
                done: true, reason: "error", n_tokens: 0, ms: 0,
                tok_per_s: 0,
                detail: "spawn returned no stdout/stderr",
            } as DoneLine);
            return;
        }
        this.proc.stdout.setEncoding("utf8");
        this.proc.stdout.on("data", (chunk: string) => this.onChunk(chunk));
        this.proc.stderr.on("data", (chunk: Buffer) => {
            // Forward stderr to the extension log, never to the chat UI.
            console.error(`[eosllm-cli] ${chunk.toString().trimEnd()}`);
        });
        this.proc.on("close", (code) => {
            if (code !== 0 && code !== null) {
                this.emit("done", {
                    done: true,
                    reason: "error",
                    n_tokens: 0,
                    ms: 0,
                    tok_per_s: 0,
                    detail: `exit ${code}`,
                } as DoneLine);
            }
        });
        this.proc.on("error", (err: Error) => {
            this.emit("done", {
                done: true,
                reason: "error",
                n_tokens: 0,
                ms: 0,
                tok_per_s: 0,
                detail: err.message,
            } as DoneLine);
        });
    }

    cancel(): void {
        if (this.proc !== null) {
            this.proc.kill("SIGTERM");
            this.proc = null;
        }
    }

    private onChunk(chunk: string): void {
        this.buf += chunk;
        let nl: number;
        // eslint-disable-next-line no-cond-assign
        while ((nl = this.buf.indexOf("\n")) >= 0) {
            const line = this.buf.slice(0, nl).trim();
            this.buf = this.buf.slice(nl + 1);
            if (line.length === 0) { continue; }
            try {
                const obj = JSON.parse(line);
                if (obj && obj.done === true) {
                    this.emit("done", obj as DoneLine);
                } else if (obj && typeof obj.t === "string") {
                    this.emit("token", obj as TokenLine);
                }
            } catch (e) {
                console.error(`[eosllm] malformed JSONL: ${line}`);
            }
        }
    }
}

// ---------------------------------------------------------------------
// Binary discovery
// ---------------------------------------------------------------------

function defaultBinDir(ctx: vscode.ExtensionContext): string {
    return path.join(ctx.globalStorageUri.fsPath, "bin");
}

function resolveCliPath(ctx: vscode.ExtensionContext): string | null {
    const cfg = vscode.workspace.getConfiguration("eosllm");
    const overrideDir = (cfg.get<string>("binDir") || "").trim();
    const dir = overrideDir.length > 0 ? overrideDir : defaultBinDir(ctx);
    const exe = process.platform === "win32" ? "eosllm-cli.exe" : "eosllm-cli";
    const full = path.join(dir, exe);
    return fs.existsSync(full) ? full : null;
}

// ---------------------------------------------------------------------
// Webview chat view
// ---------------------------------------------------------------------

class ChatViewProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = "eosllm.chatView";
    private view: vscode.WebviewView | null = null;
    private session: EosllmSession | null = null;
    private status: vscode.StatusBarItem;

    constructor(private readonly ctx: vscode.ExtensionContext) {
        this.status = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right, 50);
        this.status.text = "$(comment-discussion) eosllm: idle";
        this.status.show();
        ctx.subscriptions.push(this.status);
    }

    resolveWebviewView(view: vscode.WebviewView): void {
        this.view = view;
        view.webview.options = {
            enableScripts: true,
            localResourceRoots: [
                vscode.Uri.joinPath(this.ctx.extensionUri, "src", "webview"),
            ],
        };
        view.webview.html = this.loadHtml();
        view.webview.onDidReceiveMessage((msg) => this.onMessage(msg));
    }

    private loadHtml(): string {
        const htmlPath = path.join(
            this.ctx.extensionPath, "src", "webview", "chat.html");
        try {
            return fs.readFileSync(htmlPath, "utf8");
        } catch {
            return "<html><body><p>chat.html missing</p></body></html>";
        }
    }

    private postToken(text: string): void {
        if (this.view !== null) {
            void this.view.webview.postMessage({ kind: "token", text });
        }
    }

    private postDone(d: DoneLine): void {
        if (this.view !== null) {
            void this.view.webview.postMessage({ kind: "done", ...d });
        }
    }

    private onMessage(msg: { kind: string; prompt?: string }): void {
        if (msg.kind !== "send" || typeof msg.prompt !== "string") { return; }
        const cfg      = vscode.workspace.getConfiguration("eosllm");
        const cliPath  = resolveCliPath(this.ctx);
        const modelPath = (cfg.get<string>("modelPath") || "").trim();
        const maxTokens = cfg.get<number>("maxTokens") ?? 256;

        if (cliPath === null) {
            void vscode.window.showErrorMessage(
                "eosllm-cli binary not found. Run `npm run postinstall` " +
                "or set eosllm.binDir in settings.");
            return;
        }
        if (modelPath.length === 0 || !fs.existsSync(modelPath)) {
            void vscode.window.showErrorMessage(
                "Set `eosllm.modelPath` to an existing GGUF file.");
            return;
        }

        if (this.session !== null) { this.session.cancel(); }

        const t0 = Date.now();
        let nTokens = 0;
        const sess = new EosllmSession(cliPath, modelPath, maxTokens);
        this.session = sess;
        this.status.text = "$(loading~spin) eosllm: streaming…";

        sess.on("token", (tok: TokenLine) => {
            nTokens++;
            this.postToken(tok.t);
            const dt = (Date.now() - t0) / 1000;
            const tps = dt > 0 ? Math.round(nTokens / dt) : 0;
            this.status.text =
                `$(loading~spin) eosllm: ${nTokens} tok (${tps} tok/s)`;
        });
        sess.on("done", (d: DoneLine) => {
            this.postDone(d);
            const dt = (Date.now() - t0) / 1000;
            const tps = dt > 0 ? Math.round(nTokens / dt) : 0;
            this.status.text =
                `$(check) eosllm: ${d.reason}, ${nTokens} tok @ ${tps} tok/s`;
            this.session = null;
        });
        sess.start(msg.prompt);
    }
}

// ---------------------------------------------------------------------
// Command implementations
// ---------------------------------------------------------------------

async function runPromptCommand(ctx: vscode.ExtensionContext): Promise<void> {
    const prompt = await vscode.window.showInputBox({
        prompt:      "eosllm prompt",
        placeHolder: "Ask anything…",
    });
    if (prompt === undefined || prompt.length === 0) { return; }

    const cfg       = vscode.workspace.getConfiguration("eosllm");
    const cliPath   = resolveCliPath(ctx);
    const modelPath = (cfg.get<string>("modelPath") || "").trim();
    const maxTokens = cfg.get<number>("maxTokens") ?? 256;

    if (cliPath === null) {
        void vscode.window.showErrorMessage(
            "eosllm-cli binary not found.");
        return;
    }
    if (modelPath.length === 0 || !fs.existsSync(modelPath)) {
        void vscode.window.showErrorMessage(
            "Set `eosllm.modelPath` first.");
        return;
    }

    const out = vscode.window.createOutputChannel("eosllm");
    out.show(true);
    out.appendLine(`> ${prompt}`);

    const sess = new EosllmSession(cliPath, modelPath, maxTokens);
    sess.on("token", (tok: TokenLine) => out.append(tok.t));
    sess.on("done",  (d: DoneLine) => {
        out.appendLine("");
        out.appendLine(`[done: ${d.reason}, ${d.n_tokens} tokens]`);
    });
    sess.start(prompt);
}

// ---------------------------------------------------------------------
// activate / deactivate
// ---------------------------------------------------------------------

export function activate(ctx: vscode.ExtensionContext): void {
    const provider = new ChatViewProvider(ctx);
    ctx.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            ChatViewProvider.viewType, provider),
        vscode.commands.registerCommand("eosllm.runPrompt", () =>
            runPromptCommand(ctx)),
        vscode.commands.registerCommand("eosllm.openChat", () =>
            vscode.commands.executeCommand(
                `workbench.view.extension.eosllm`)),
    );

    // Sanity log: where will we look for the binary?
    const cliPath = resolveCliPath(ctx);
    if (cliPath === null) {
        const dir = defaultBinDir(ctx);
        console.warn(
            `[eosllm] no binary at ${dir}. Run scripts/postinstall.js to fetch one.`);
    } else {
        console.log(`[eosllm] using binary at ${cliPath}`);
    }
    void os; // imported for platform-specific extensions; no-op today
}

export function deactivate(): void { /* no-op */ }
