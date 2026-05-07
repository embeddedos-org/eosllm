/*
 * vscode-extension/scripts/postinstall.js
 *
 * Downloads the platform-matching eosllm-cli binary from the engine's
 * GitHub Releases page and unpacks it into the extension's binDir.
 *
 * Runs at:
 *   1. `npm install` (development) — places binaries under ./bin so
 *      F5 development host finds them.
 *   2. First launch of the published extension — VS Code calls this
 *      via "scripts.postinstall" when the .vsix is unpacked. The
 *      target directory is the extension's globalStorage path; we
 *      detect that via the env var the extension exports on activate
 *      ($EOSLLM_BIN_DIR), or fall back to ./bin if not set.
 *
 * Pulls the *latest* engine release tag matching `v*` from
 * https://api.github.com/repos/embeddedos-org/eosllm/releases.
 *
 * Pure node (no third-party deps) so the published extension stays
 * tiny.
 */

"use strict";

const fs    = require("fs");
const path  = require("path");
const os    = require("os");
const https = require("https");
const { execSync } = require("child_process");
const zlib  = require("zlib");

const REPO    = "embeddedos-org/eosllm";
const UA      = "eosllm-vscode/0.1.0";

function detectPlatformAsset() {
    const p = process.platform;
    const a = process.arch;
    if (p === "linux"  && a === "x64") {
        return { suffix: "linux-x86_64.tar.gz",  ext: ".tar.gz" };
    }
    if (p === "linux"  && a === "arm64") {
        return { suffix: "linux-aarch64.tar.gz", ext: ".tar.gz" };
    }
    if (p === "darwin") {
        return { suffix: "macos-universal.tar.gz", ext: ".tar.gz" };
    }
    if (p === "win32"  && a === "x64") {
        return { suffix: "windows-x86_64.zip", ext: ".zip" };
    }
    throw new Error(`unsupported platform/arch: ${p}/${a}`);
}

function httpGet(url, headers) {
    return new Promise((resolve, reject) => {
        const opts = { headers: { "User-Agent": UA, ...headers } };
        https.get(url, opts, (res) => {
            // Follow one level of redirect (GitHub asset → S3).
            if (res.statusCode === 301 || res.statusCode === 302) {
                resolve(httpGet(res.headers.location, headers));
                return;
            }
            if ((res.statusCode || 0) >= 400) {
                reject(new Error(`GET ${url}: ${res.statusCode}`));
                return;
            }
            const chunks = [];
            res.on("data", (c) => chunks.push(c));
            res.on("end", () => resolve(Buffer.concat(chunks)));
        }).on("error", reject);
    });
}

async function main() {
    const asset = detectPlatformAsset();

    // Resolve binDir. Prefer the env var the extension exports; else
    // ./bin (development).
    const binDir = process.env.EOSLLM_BIN_DIR
        || path.join(__dirname, "..", "bin");
    fs.mkdirSync(binDir, { recursive: true });

    // 1. Pick the newest release matching v*.
    console.log(`[eosllm] querying ${REPO} releases…`);
    const releasesJson = await httpGet(
        `https://api.github.com/repos/${REPO}/releases?per_page=10`);
    const releases = JSON.parse(releasesJson.toString("utf8"));
    const latest = releases.find((r) => /^v\d/.test(r.tag_name) && !r.draft);
    if (!latest) {
        console.error(`[eosllm] no eligible release tag found`);
        process.exit(1);
    }
    console.log(`[eosllm] latest release: ${latest.tag_name}`);

    // 2. Find the platform-matching asset.
    const assetUrl = (latest.assets || [])
        .map((a) => a.browser_download_url)
        .find((u) => u.endsWith(asset.suffix));
    if (!assetUrl) {
        console.error(
            `[eosllm] release ${latest.tag_name} has no asset matching ${asset.suffix}`);
        process.exit(1);
    }
    console.log(`[eosllm] downloading ${assetUrl}`);

    // 3. Download and stage in a temp file.
    const data = await httpGet(assetUrl);
    const stagePath = path.join(os.tmpdir(),
        `eosllm-asset-${Date.now()}${asset.ext}`);
    fs.writeFileSync(stagePath, data);

    // 4. Extract. tar.gz handled via system `tar`; .zip via system
    //    `unzip` (windows-latest ships PowerShell `Expand-Archive`).
    const extractDir = path.join(os.tmpdir(),
        `eosllm-extract-${Date.now()}`);
    fs.mkdirSync(extractDir, { recursive: true });

    if (asset.ext === ".tar.gz") {
        execSync(`tar xzf "${stagePath}" -C "${extractDir}"`,
                 { stdio: "inherit" });
    } else if (asset.ext === ".zip") {
        if (process.platform === "win32") {
            execSync(`powershell -NoProfile -Command "Expand-Archive '${stagePath}' '${extractDir}'"`,
                     { stdio: "inherit" });
        } else {
            execSync(`unzip -o "${stagePath}" -d "${extractDir}"`,
                     { stdio: "inherit" });
        }
    }

    // 5. Locate the bin/ folder inside the extracted tree and copy
    //    every binary into binDir.
    function findBinDir(start) {
        const entries = fs.readdirSync(start, { withFileTypes: true });
        for (const e of entries) {
            const full = path.join(start, e.name);
            if (e.isDirectory()) {
                if (e.name === "bin") return full;
                const sub = findBinDir(full);
                if (sub !== null) return sub;
            }
        }
        return null;
    }
    const srcBin = findBinDir(extractDir);
    if (!srcBin) {
        console.error(`[eosllm] no bin/ directory inside ${extractDir}`);
        process.exit(1);
    }

    for (const fname of fs.readdirSync(srcBin)) {
        const src = path.join(srcBin, fname);
        const dst = path.join(binDir, fname);
        fs.copyFileSync(src, dst);
        if (process.platform !== "win32") {
            fs.chmodSync(dst, 0o755);
        }
    }
    console.log(`[eosllm] installed binaries to ${binDir}`);

    // 6. Cleanup.
    try { fs.unlinkSync(stagePath); } catch (_e) { /* best-effort */ }
    try { fs.rmSync(extractDir, { recursive: true, force: true }); }
    catch (_e) { /* best-effort */ }
}

main().catch((err) => {
    console.error(`[eosllm] postinstall failed: ${err.message}`);
    process.exit(1);
});
