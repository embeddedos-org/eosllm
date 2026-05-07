/*
 * tools/eosllm-server/main.c — single-binary local HTTP/SSE daemon.
 *
 *   eosllm-server [--port N] [--bind 127.0.0.1] [--allow-origin <orig>]
 *
 * Routes:
 *   GET  /healthz   → "ok\n"            (200)
 *   GET  /caps      → JSON, mirrors `eosllm-cli --caps` (200)
 *   POST /generate  → text/event-stream
 *                     Body JSON: { "prompt": "...", "max_tokens": N,
 *                                  "model": "<path>" (optional) }
 *                     Response stream: one `event: token` per token,
 *                     then a final `event: done` carrying summary.
 *
 * Design:
 *   - Pure C99, zero third-party deps (matches the engine's policy).
 *   - Hand-rolled HTTP/1.1 parser + SSE encoder. Single-threaded
 *     select() loop, one connection at a time (the engine itself is
 *     single-threaded in default builds).
 *   - Loopback-bind (127.0.0.1) by default. CORS default-deny;
 *     --allow-origin sets a specific Access-Control-Allow-Origin.
 *   - Same engine wiring as eosllm-cli (calls eos_init_defaults +
 *     eos_session_*); no IPC, no fork.
 *   - On Win32 we wrap socket(), bind(), etc. in <winsock2.h>.
 *
 * Security:
 *   - Default bind is 127.0.0.1 to avoid casual LAN exposure. The
 *     --bind flag is provided for users who run inside a sandboxed
 *     container with localhost-only networking.
 *   - Request body capped at 64 KiB; rejects with 413 otherwise.
 *   - JSON parser is a hand-rolled minimal subset that accepts only
 *     the documented schema; unknown keys are ignored, malformed
 *     input rejected with 400.
 *   - No path traversal: --model paths are passed through to
 *     eos_model_open which only opens the path it's given.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "eosllm/eosllm.h"
#include "eosllm/model.h"
#include "eosllm/os.h"

/* ------------------------------------------------------------------ */
/* Cross-platform socket includes                                      */
/* ------------------------------------------------------------------ */
#if defined(_WIN32) || defined(_WIN64)
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32.lib")
   typedef int socklen_t;
#  define SHUT_RDWR SD_BOTH
#  define close_socket(s) closesocket(s)
   static int socket_init(void) {
       WSADATA wsa;
       return WSAStartup(MAKEWORD(2, 2), &wsa) == 0 ? 0 : -1;
   }
   static void socket_shutdown(void) { WSACleanup(); }
#else
#  include <sys/socket.h>
#  include <netinet/in.h>
#  include <arpa/inet.h>
#  include <unistd.h>
#  include <sys/select.h>
#  include <signal.h>
#  define close_socket(s) close(s)
   typedef int SOCKET;
#  define INVALID_SOCKET (-1)
#  define SOCKET_ERROR   (-1)
   static int socket_init(void) {
       /* Don't get killed by SIGPIPE if the client hangs up mid-stream. */
       signal(SIGPIPE, SIG_IGN);
       return 0;
   }
   static void socket_shutdown(void) {}
#endif

/* ------------------------------------------------------------------ */
/* Static config + helpers                                             */
/* ------------------------------------------------------------------ */
#define HTTP_MAX_REQ_BYTES   (64u * 1024u)
#define HTTP_RECV_CHUNK      4096u
#define EOSLLM_SERVER_VERSION "0.1.0"

static const char *g_allow_origin = NULL;   /* CORS allow-list */

static void usage(void) {
    fprintf(stderr,
        "Usage: eosllm-server [--port N] [--bind ADDR] [--allow-origin ORIG]\n"
        "\n"
        "Defaults: --port 7777 --bind 127.0.0.1\n"
        "\n"
        "Routes:\n"
        "  GET  /healthz   200, body=\"ok\\n\"\n"
        "  GET  /caps      200, JSON capabilities\n"
        "  POST /generate  Server-Sent Events token stream\n"
        "                  body: { \"prompt\": str, \"max_tokens\": int,\n"
        "                          \"model\": str (optional) }\n");
}

/* Send all bytes; returns 0 on success, -1 on partial write / error. */
static int send_all(SOCKET s, const char *buf, size_t len) {
    size_t off = 0;
    while (off < len) {
        int n = send(s, buf + off, (int)(len - off), 0);
        if (n <= 0) return -1;
        off += (size_t)n;
    }
    return 0;
}

static int send_str(SOCKET s, const char *str) {
    return send_all(s, str, strlen(str));
}

/* Append a CORS Access-Control-Allow-Origin header iff configured. */
static void append_cors(char *buf, size_t cap) {
    if (g_allow_origin == NULL || g_allow_origin[0] == '\0') return;
    /* Buffer is owned by caller; we tack on. Caller sized for headroom. */
    size_t l = strlen(buf);
    int n = snprintf(buf + l, cap - l,
                     "Access-Control-Allow-Origin: %s\r\n",
                     g_allow_origin);
    if (n < 0 || (size_t)n >= cap - l) buf[l] = '\0'; /* drop on overflow */
}

/* ------------------------------------------------------------------ */
/* Tiny JSON helpers — find key, copy string value or read int.        */
/* ------------------------------------------------------------------ */

/* Locate a "key" in a JSON object, return pointer just past the colon
 * or NULL. Naive but sufficient for the documented schema; unknown
 * structure is rejected by the consumer below. */
static const char *json_find_key(const char *body, size_t len, const char *key) {
    size_t klen = strlen(key);
    size_t i;
    if (body == NULL || len == 0) return NULL;
    for (i = 0; i + klen + 2 < len; ++i) {
        if (body[i] != '"') continue;
        if (memcmp(body + i + 1, key, klen) != 0) continue;
        if (body[i + 1 + klen] != '"') continue;
        /* Skip past key. Find first ':'. */
        size_t j = i + 2 + klen;
        while (j < len && body[j] != ':') {
            if (body[j] != ' ' && body[j] != '\t') return NULL;
            j++;
        }
        if (j >= len) return NULL;
        j++;
        while (j < len && (body[j] == ' ' || body[j] == '\t' || body[j] == '\n')) j++;
        return body + j;
    }
    return NULL;
}

/* Parse a JSON string starting at *cursor (must point at the leading
 * '"'). Writes up to cap-1 bytes into out, NUL-terminates. Returns
 * the length, or -1 on error. */
static int json_parse_string(const char *cursor, char *out, size_t cap) {
    size_t i = 0, w = 0;
    if (cursor == NULL || cap == 0) return -1;
    if (cursor[0] != '"') return -1;
    i = 1;
    while (cursor[i] != '\0' && cursor[i] != '"' && w < cap - 1) {
        if (cursor[i] == '\\' && cursor[i + 1] != '\0') {
            char esc = cursor[i + 1];
            char ch  = esc;
            switch (esc) {
                case 'n': ch = '\n'; break;
                case 't': ch = '\t'; break;
                case 'r': ch = '\r'; break;
                case '"': ch = '"';  break;
                case '\\': ch = '\\'; break;
                case '/': ch = '/';  break;
                default: break; /* leave \uXXXX, \b, \f raw — unused */
            }
            out[w++] = ch;
            i += 2;
        } else {
            out[w++] = cursor[i++];
        }
    }
    out[w] = '\0';
    return cursor[i] == '"' ? (int)w : -1;
}

static int json_parse_int(const char *cursor) {
    if (cursor == NULL) return -1;
    return atoi(cursor);
}

/* ------------------------------------------------------------------ */
/* Caps JSON renderer — mirrors `eosllm-cli --caps` output             */
/* ------------------------------------------------------------------ */

static void render_caps_json(char *out, size_t cap) {
    eos_caps_t c;
    if (eos_caps(&c) != EOS_OK) {
        snprintf(out, cap, "{ \"error\": \"eos_caps failed\" }\n");
        return;
    }
    snprintf(out, cap,
        "{\n"
        "  \"library_version\": \"%s\",\n"
        "  \"abi_version\": %d,\n"
        "  \"server_version\": \"%s\",\n"
        "  \"compile_time\": {\n"
        "    \"posix\": %u, \"win32\": %u, \"zephyr\": %u, "
        "\"freertos\": %u, \"baremetal\": %u,\n"
        "    \"threads\": %u,\n"
        "    \"kernels\": { \"scalar\": %u, \"avx2\": %u, \"avx512\": %u, "
        "\"neon\": %u, \"sve\": %u, \"rvv\": %u, \"hvx\": %u, \"npu\": %u },\n"
        "    \"quants\": { \"q8_0\": %u, \"q4_k\": %u, \"q2_k\": %u, "
        "\"q1_58\": %u, \"mixed\": %u, \"calibrated\": %u },\n"
        "    \"modalities\": { \"text\": %u, \"vision\": %u, \"audio\": %u },\n"
        "    \"tokenizers\": { \"bpe\": %u, \"spm\": %u },\n"
        "    \"formats\": { \"eosm\": %u, \"gguf\": %u },\n"
        "    \"schedulers\": { \"greedy\": %u, \"deadline\": %u, \"batched\": %u }\n"
        "  },\n"
        "  \"runtime_bits\": {\n"
        "    \"avx2\":   %u, \"avx512\": %u, \"neon\":   %u, \"sve\":    %u,\n"
        "    \"rvv\":    %u, \"hvx\":    %u, \"npu\":    %u\n"
        "  }\n"
        "}\n",
        eos_version_string(), eos_abi_version(), EOSLLM_SERVER_VERSION,
        c.have_posix,
#if defined(_WIN32) || defined(_WIN64)
        1u,
#else
        0u,
#endif
        c.have_zephyr, c.have_freertos, c.have_baremetal,
        c.have_threads,
        c.have_k_scalar, c.have_k_avx2, c.have_k_avx512, c.have_k_neon,
        c.have_k_sve, c.have_k_rvv, c.have_k_hvx, c.have_k_npu,
        c.have_q_q8_0, c.have_q_q4_k, c.have_q_q2_k,
        c.have_q_q1_58, c.have_q_mixed, c.have_q_calibrtd,
        c.have_m_text, c.have_m_vision, c.have_m_audio,
        c.have_t_bpe, c.have_t_spm,
        c.have_f_eosm, c.have_f_gguf,
        c.have_s_greedy, c.have_s_deadline, c.have_s_batched,
        (c.runtime_bits & EOS_CAPS_RT_AVX2)   ? 1u : 0u,
        (c.runtime_bits & EOS_CAPS_RT_AVX512) ? 1u : 0u,
        (c.runtime_bits & EOS_CAPS_RT_NEON)   ? 1u : 0u,
        (c.runtime_bits & EOS_CAPS_RT_SVE)    ? 1u : 0u,
        (c.runtime_bits & EOS_CAPS_RT_RVV)    ? 1u : 0u,
        (c.runtime_bits & EOS_CAPS_RT_HVX)    ? 1u : 0u,
        (c.runtime_bits & EOS_CAPS_RT_NPU)    ? 1u : 0u);
}

/* ------------------------------------------------------------------ */
/* HTTP response shapers                                               */
/* ------------------------------------------------------------------ */

static void respond_text(SOCKET s, int status, const char *reason,
                         const char *content_type, const char *body) {
    char headers[1024];
    size_t blen = body ? strlen(body) : 0;
    snprintf(headers, sizeof(headers),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n",
        status, reason, content_type, blen);
    append_cors(headers, sizeof(headers));
    /* Two CRLFs total (header sep + after status/header block ends) */
    {
        size_t hl = strlen(headers);
        if (hl + 2 < sizeof(headers)) {
            headers[hl]   = '\r';
            headers[hl+1] = '\n';
            headers[hl+2] = '\0';
        }
    }
    (void)send_str(s, headers);
    if (body && blen > 0) (void)send_all(s, body, blen);
}

static void respond_404(SOCKET s) {
    respond_text(s, 404, "Not Found", "text/plain",
                 "404 not found\n");
}
static void respond_400(SOCKET s, const char *why) {
    char body[256];
    snprintf(body, sizeof(body), "400 bad request: %s\n", why);
    respond_text(s, 400, "Bad Request", "text/plain", body);
}
static void respond_405(SOCKET s) {
    respond_text(s, 405, "Method Not Allowed", "text/plain",
                 "405 method not allowed\n");
}
static void respond_413(SOCKET s) {
    respond_text(s, 413, "Payload Too Large", "text/plain",
                 "413 payload too large\n");
}

/* Begin SSE response: send header then return; caller streams events. */
static int begin_sse(SOCKET s) {
    char headers[1024];
    snprintf(headers, sizeof(headers),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n"
        "X-Accel-Buffering: no\r\n");
    append_cors(headers, sizeof(headers));
    {
        size_t hl = strlen(headers);
        if (hl + 2 < sizeof(headers)) {
            headers[hl]   = '\r';
            headers[hl+1] = '\n';
            headers[hl+2] = '\0';
        }
    }
    return send_str(s, headers);
}

static int sse_event(SOCKET s, const char *event, const char *data) {
    char buf[1024];
    /* "event: <event>\ndata: <data>\n\n"; <data> must not contain raw
     * \n (we json-encode in the caller). */
    int n = snprintf(buf, sizeof(buf),
                     "event: %s\ndata: %s\n\n", event, data);
    if (n < 0 || (size_t)n >= sizeof(buf)) return -1;
    return send_all(s, buf, (size_t)n);
}

/* ------------------------------------------------------------------ */
/* /generate handler                                                   */
/* ------------------------------------------------------------------ */

static void handle_generate(SOCKET s, const char *body, size_t blen) {
    char prompt[8192];
    char model_path[2048];
    int  max_tokens = 64;
    eos_status_t st;
    eos_model_t   *m  = NULL;
    eos_session_t *ss = NULL;
    eos_session_params_t params;
    const char *cursor;

    /* Parse prompt (required). */
    cursor = json_find_key(body, blen, "prompt");
    if (cursor == NULL || json_parse_string(cursor, prompt, sizeof(prompt)) < 0) {
        respond_400(s, "missing or malformed \"prompt\"");
        return;
    }

    /* Parse max_tokens (optional). */
    cursor = json_find_key(body, blen, "max_tokens");
    if (cursor != NULL) {
        max_tokens = json_parse_int(cursor);
        if (max_tokens <= 0 || max_tokens > 8192) max_tokens = 64;
    }

    /* Parse model path (optional). When absent, run the smoke flow:
     * a NULL-model session that exercises feed/step EOS_E_UNSUPPORTED
     * paths, useful for protocol smoke without a real .gguf at hand. */
    cursor = json_find_key(body, blen, "model");
    model_path[0] = '\0';
    if (cursor != NULL) {
        if (json_parse_string(cursor, model_path, sizeof(model_path)) < 0) {
            respond_400(s, "malformed \"model\"");
            return;
        }
    }

    if (begin_sse(s) != 0) return;

    if (model_path[0] != '\0') {
        st = eos_model_open(model_path, &m);
        if (st != EOS_OK || m == NULL) {
            char msg[512];
            snprintf(msg, sizeof(msg),
                     "{\"reason\":\"error\",\"detail\":\"model_open: %s\"}",
                     eos_status_str(st));
            (void)sse_event(s, "done", msg);
            return;
        }
    }

    memset(&params, 0, sizeof(params));
    params.max_context = (uint32_t)(max_tokens + 256);
    st = eos_session_open(m, &params, &ss);
    if (st != EOS_OK || ss == NULL) {
        char msg[512];
        const char *e = eos_last_error();
        snprintf(msg, sizeof(msg),
                 "{\"reason\":\"error\",\"detail\":\"session_open: %s%s%s\"}",
                 eos_status_str(st), e ? " (" : "", e ? e : "");
        if (e) {
            size_t l = strlen(msg);
            if (l + 2 < sizeof(msg)) { msg[l-1] = ')'; msg[l] = '"'; msg[l+1] = '}'; msg[l+2] = '\0'; }
        }
        (void)sse_event(s, "done", msg);
        if (m) eos_model_close(m);
        return;
    }

    st = eos_session_feed(ss, EOS_MODALITY_TEXT, prompt, strlen(prompt));
    if (st != EOS_OK) {
        char msg[512];
        snprintf(msg, sizeof(msg),
                 "{\"reason\":\"error\",\"detail\":\"feed: %s\"}",
                 eos_status_str(st));
        (void)sse_event(s, "done", msg);
        eos_session_close(ss);
        if (m) eos_model_close(m);
        return;
    }

    {
        int i;
        int n_emitted = 0;
        uint64_t t0 = 0;
        const eos_os_shim_t *os; /* unused but reminder */
        (void)os;
        for (i = 0; i < max_tokens; ++i) {
            uint32_t tok;
            char     buf[64];
            size_t   len = sizeof(buf);
            char     ev[256];
            st = eos_session_step(ss, &tok);
            if (st != EOS_OK) break;
            st = eos_session_decode(ss, &tok, 1, buf, &len);
            if (st == EOS_OK && len <= sizeof(buf)) {
                /* Encode token as a small JSON object. We only need to
                 * escape ", \, and newline for strict JSON. */
                char enc[128];
                size_t w = 0, k;
                for (k = 0; k < len && w < sizeof(enc) - 6; ++k) {
                    unsigned char c = (unsigned char)buf[k];
                    if (c == '"' || c == '\\') { enc[w++] = '\\'; enc[w++] = (char)c; }
                    else if (c == '\n') { enc[w++] = '\\'; enc[w++] = 'n'; }
                    else if (c == '\r') { enc[w++] = '\\'; enc[w++] = 'r'; }
                    else if (c == '\t') { enc[w++] = '\\'; enc[w++] = 't'; }
                    else if (c < 0x20)  { /* drop other control bytes */ }
                    else                { enc[w++] = (char)c; }
                }
                enc[w] = '\0';
                snprintf(ev, sizeof(ev),
                         "{\"t\":\"%s\",\"i\":%u}", enc, (unsigned)tok);
                if (sse_event(s, "token", ev) != 0) break;
                n_emitted++;
            }
        }
        (void)t0;
        {
            char msg[256];
            snprintf(msg, sizeof(msg),
                     "{\"reason\":\"%s\",\"n_tokens\":%d}",
                     (st == EOS_OK ? "max" : eos_status_str(st)),
                     n_emitted);
            (void)sse_event(s, "done", msg);
        }
    }

    eos_session_close(ss);
    if (m) eos_model_close(m);
}

/* ------------------------------------------------------------------ */
/* Per-request loop                                                    */
/* ------------------------------------------------------------------ */

static void serve_one(SOCKET cli) {
    static char req[HTTP_MAX_REQ_BYTES + 1];
    size_t total = 0;
    size_t header_end = 0;
    int    saw_header_end = 0;

    /* Read until end of headers (CRLFCRLF) or until we exceed cap. */
    for (;;) {
        int n;
        if (total >= HTTP_MAX_REQ_BYTES) { respond_413(cli); return; }
        n = recv(cli, req + total,
                 (int)(HTTP_MAX_REQ_BYTES - total), 0);
        if (n <= 0) return;
        total += (size_t)n;
        req[total] = '\0';
        {
            char *p = strstr(req, "\r\n\r\n");
            if (p != NULL) {
                header_end    = (size_t)(p - req) + 4;
                saw_header_end = 1;
                break;
            }
        }
    }

    if (!saw_header_end) { respond_400(cli, "no header terminator"); return; }

    /* Parse the request line. */
    {
        char method[16];
        char path[512];
        int  i;
        const char *p = req;
        for (i = 0; i < (int)sizeof(method) - 1 && *p && *p != ' '; ++i) {
            method[i] = *p++;
        }
        method[i] = '\0';
        if (*p != ' ') { respond_400(cli, "no method"); return; }
        p++;
        for (i = 0; i < (int)sizeof(path) - 1 && *p && *p != ' '; ++i) {
            path[i] = *p++;
        }
        path[i] = '\0';
        if (*p != ' ') { respond_400(cli, "no path"); return; }

        if (strcmp(method, "GET") == 0 && strcmp(path, "/healthz") == 0) {
            respond_text(cli, 200, "OK", "text/plain", "ok\n");
            return;
        }
        if (strcmp(method, "GET") == 0 && strcmp(path, "/caps") == 0) {
            char body[4096];
            render_caps_json(body, sizeof(body));
            respond_text(cli, 200, "OK", "application/json", body);
            return;
        }
        if (strcmp(method, "OPTIONS") == 0) {
            /* CORS preflight. */
            char headers[1024];
            snprintf(headers, sizeof(headers),
                "HTTP/1.1 204 No Content\r\n"
                "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                "Access-Control-Allow-Headers: Content-Type\r\n"
                "Content-Length: 0\r\n"
                "Connection: close\r\n");
            append_cors(headers, sizeof(headers));
            {
                size_t hl = strlen(headers);
                if (hl + 2 < sizeof(headers)) {
                    headers[hl]   = '\r';
                    headers[hl+1] = '\n';
                    headers[hl+2] = '\0';
                }
            }
            (void)send_str(cli, headers);
            return;
        }
        if (strcmp(method, "POST") == 0 && strcmp(path, "/generate") == 0) {
            /* Locate Content-Length to bound the body read. */
            const char *cl = strstr(req, "Content-Length:");
            size_t body_len = 0;
            if (cl != NULL) body_len = (size_t)atoi(cl + 15);
            /* If body wasn't fully received, keep reading. */
            while (total - header_end < body_len) {
                int n;
                if (total >= HTTP_MAX_REQ_BYTES) { respond_413(cli); return; }
                n = recv(cli, req + total,
                         (int)(HTTP_MAX_REQ_BYTES - total), 0);
                if (n <= 0) break;
                total += (size_t)n;
                req[total] = '\0';
            }
            if (body_len == 0) { respond_400(cli, "empty body"); return; }
            handle_generate(cli, req + header_end, body_len);
            return;
        }
        if (strcmp(method, "GET") == 0 || strcmp(method, "POST") == 0) {
            respond_404(cli);
            return;
        }
        respond_405(cli);
    }
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    int port = 7777;
    const char *bind_addr = "127.0.0.1";
    int i;
    SOCKET listener;
    struct sockaddr_in addr;
    int yes = 1;
    eos_status_t st;

    for (i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--port")          && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bind")          && i + 1 < argc) bind_addr = argv[++i];
        else if (!strcmp(argv[i], "--allow-origin")  && i + 1 < argc) g_allow_origin = argv[++i];
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
            { usage(); return 0; }
        else if (!strcmp(argv[i], "--version") || !strcmp(argv[i], "-V"))
            { printf("eosllm-server %s engine=%s abi=%d\n",
                     EOSLLM_SERVER_VERSION, eos_version_string(),
                     eos_abi_version()); return 0; }
        else { usage(); return 2; }
    }

    if (port <= 0 || port > 65535) { fprintf(stderr, "bad --port\n"); return 2; }

    if (socket_init() != 0) {
        fprintf(stderr, "socket_init failed\n");
        return 1;
    }

    st = eos_init_defaults();
    if (st != EOS_OK) {
        fprintf(stderr, "eos_init_defaults: %s\n", eos_status_str(st));
        socket_shutdown();
        return 1;
    }

    listener = socket(AF_INET, SOCK_STREAM, 0);
    if (listener == INVALID_SOCKET) {
        fprintf(stderr, "socket failed\n");
        socket_shutdown();
        return 1;
    }
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR,
               (const char *)&yes, sizeof(yes));

    memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons((unsigned short)port);
    if (inet_pton(AF_INET, bind_addr, &addr.sin_addr) != 1) {
        fprintf(stderr, "bad --bind address: %s\n", bind_addr);
        close_socket(listener);
        socket_shutdown();
        return 2;
    }

    if (bind(listener, (struct sockaddr *)&addr, sizeof(addr)) == SOCKET_ERROR) {
        fprintf(stderr, "bind %s:%d failed\n", bind_addr, port);
        close_socket(listener);
        socket_shutdown();
        return 1;
    }
    if (listen(listener, 8) == SOCKET_ERROR) {
        fprintf(stderr, "listen failed\n");
        close_socket(listener);
        socket_shutdown();
        return 1;
    }

    fprintf(stdout, "eosllm-server %s listening on http://%s:%d\n",
            EOSLLM_SERVER_VERSION, bind_addr, port);
    fflush(stdout);

    for (;;) {
        struct sockaddr_in cli_addr;
        socklen_t cl = (socklen_t)sizeof(cli_addr);
        SOCKET cli = accept(listener, (struct sockaddr *)&cli_addr, &cl);
        if (cli == INVALID_SOCKET) continue;
        serve_one(cli);
        shutdown(cli, SHUT_RDWR);
        close_socket(cli);
    }

    /* unreachable */
    close_socket(listener);
    socket_shutdown();
    return 0;
}
