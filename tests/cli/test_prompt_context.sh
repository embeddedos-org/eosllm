#!/bin/bash
# The first new token must be predicted from the prompt exactly as
# given: each prompt token goes into the KV cache once.
#
# Writes a tiny Llama-architecture GGUF: F32, 1 layer, 1 head, n_embd 4, a
# 16-token vocabulary starting "<s>", "</s>", "a", "b". q and k are zero, so
# attention is a plain average of v over the context; v copies "a" into
# slot 2 and "b" into slot 3, and output.weight scores "c" from slot 2 and
# "d" from 0.9 x slot 3. So:
#   context "a b"   -> slot 2 == slot 3            -> "c"
#   context "a b b" -> slot 3 twice as big as slot 2 -> "d"
# Checks:
#   1. prompt "ab": the first token is "c";
#   2. prompt "a" (one token): the first token is "c";
#   3. prompt "ab" with --ctx 2: the prompt fits, so one token is produced;
#   4. the same with --n 2: the second step finds the context full, so the
#      run must end with reason "error" and exit 1 (docs/cli.md).
# Exits non-zero on any mismatch.
set -euo pipefail
cd "$(dirname "$0")/../.."

CLI=${EOSLLM_CLI:-tools/eosllm-cli/eosllm-cli}
TMPDIR_=$(mktemp -d -t eosllm_prompt_ctx.XXXXXX)
trap 'rm -rf "$TMPDIR_"' EXIT

MODEL="$TMPDIR_/ctx.gguf"
python3 - "$MODEL" << 'PY'
import struct, sys

U32, STRING, ARRAY = 4, 8, 9
GGML_F32 = 0
ALIGN = 32
n_embd, n_ff, n_ctx = 4, 4, 16
tokens = ['<s>', '</s>'] + [chr(c) for c in range(ord('a'), ord('a') + 14)]
n_vocab = len(tokens)
BOS, EOS, A, B, C, D = 0, 1, 2, 3, 4, 5

def lstr(s):
    b = s.encode()
    return struct.pack('<Q', len(b)) + b

kvs = [
    ('general.architecture', STRING, 'llama'),
    ('general.alignment', U32, ALIGN),
    ('llama.embedding_length', U32, n_embd),
    ('llama.block_count', U32, 1),
    ('llama.attention.head_count', U32, 1),
    ('llama.feed_forward_length', U32, n_ff),
    ('llama.context_length', U32, n_ctx),
    ('tokenizer.ggml.model', STRING, 'llama'),
    ('tokenizer.ggml.tokens', ARRAY, tokens),
    ('tokenizer.ggml.bos_token_id', U32, BOS),
]

def rows(n, set_rows):
    m = [[0.0] * n_embd for _ in range(n)]
    for r, v in set_rows.items():
        m[r] = v
    return [x for row in m for x in row]

e = [[1.0 if i == j else 0.0 for i in range(n_embd)] for j in range(n_embd)]
ones = [1.0] * n_embd
zeros = [0.0] * (n_embd * n_embd)
tensors = [
    # [n_embd, n_vocab] in GGUF order: one row of n_embd per token.
    ('token_embd.weight', [n_embd, n_vocab], rows(n_vocab, {A: e[0], B: e[1]})),
    ('output_norm.weight', [n_embd], ones),
    ('output.weight', [n_embd, n_vocab],
     rows(n_vocab, {C: e[2], D: [0.0, 0.0, 0.0, 0.9]})),
    ('blk.0.attn_norm.weight', [n_embd], ones),
    ('blk.0.attn_q.weight', [n_embd, n_embd], zeros),
    ('blk.0.attn_k.weight', [n_embd, n_embd], zeros),
    # v[2] = xn[0] ("a"), v[3] = xn[1] ("b").
    ('blk.0.attn_v.weight', [n_embd, n_embd], rows(n_embd, {2: e[0], 3: e[1]})),
    ('blk.0.attn_output.weight', [n_embd, n_embd], [x for row in e for x in row]),
    ('blk.0.ffn_norm.weight', [n_embd], ones),
    ('blk.0.ffn_gate.weight', [n_embd, n_ff], zeros),
    ('blk.0.ffn_up.weight', [n_embd, n_ff], zeros),
    ('blk.0.ffn_down.weight', [n_ff, n_embd], zeros),
]

buf = bytearray()
buf += struct.pack('<IIQQ', 0x46554747, 3, len(tensors), len(kvs))
for key, typ, val in kvs:
    buf += lstr(key) + struct.pack('<I', typ)
    if typ == U32:
        buf += struct.pack('<I', val)
    elif typ == STRING:
        buf += lstr(val)
    else:
        buf += struct.pack('<IQ', STRING, len(val)) + b''.join(lstr(s) for s in val)

data = bytearray()
for name, dims, vals in tensors:
    while len(data) % ALIGN:
        data += b'\x00'
    buf += lstr(name) + struct.pack('<I', len(dims))
    buf += b''.join(struct.pack('<Q', d) for d in dims)
    buf += struct.pack('<IQ', GGML_F32, len(data))
    data += struct.pack('<%df' % len(vals), *vals)
while len(buf) % ALIGN:
    buf += b'\x00'
open(sys.argv[1], 'wb').write(bytes(buf + data))
PY

fail() { echo "FAIL: $1"; echo "$2"; exit 1; }

echo "--- 1. prompt \"ab\": first token ---"
out=$("$CLI" --model "$MODEL" --prompt "ab" --n 1 --stream-jsonl)
echo "$out"
expect='{"t":"c","i":4}
{"done":true,"reason":"max","n_tokens":1,"ms":0,"tok_per_s":0}'
[ "$out" = "$expect" ] || fail "expected \"c\" (context \"a b\"); \"d\" means \"b\" was fed twice" "$out"

echo "--- 2. prompt \"a\" (one token): first token ---"
out=$("$CLI" --model "$MODEL" --prompt "a" --n 1 --stream-jsonl)
echo "$out"
expect='{"t":"c","i":4}
{"done":true,"reason":"max","n_tokens":1,"ms":0,"tok_per_s":0}'
[ "$out" = "$expect" ] || fail "expected \"c\" after a one-token prompt" "$out"

echo "--- 3. prompt \"ab\" with --ctx 2 ---"
out=$("$CLI" --model "$MODEL" --prompt "ab" --n 1 --ctx 2 --stream-jsonl)
echo "$out"
expect='{"t":"c","i":4}
{"done":true,"reason":"max","n_tokens":1,"ms":0,"tok_per_s":0}'
[ "$out" = "$expect" ] || fail "a 2-token prompt in a 2-token context must still produce a token" "$out"

echo "--- 4. prompt \"ab\" with --ctx 2 --n 2: context fills up ---"
rc=0
out=$("$CLI" --model "$MODEL" --prompt "ab" --n 2 --ctx 2 --stream-jsonl) || rc=$?
echo "$out"
echo "exit status: $rc"
expect='{"t":"c","i":4}
{"done":true,"reason":"error","n_tokens":1,"ms":0,"tok_per_s":0}'
[ "$out" = "$expect" ] || fail "expected one token, then reason \"error\"" "$out"
[ "$rc" -eq 1 ] || fail "a failed generation must exit 1, got $rc" "$out"

echo "prompt-context: OK (4 checks)"
