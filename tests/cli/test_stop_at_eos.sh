#!/bin/bash
# eosllm-cli must stop generating when the model emits its end-of-sequence
# token, and report it as reason "eos" (docs/cli.md, --stream-jsonl).
#
# Writes a tiny Llama-architecture GGUF: F32, 1 layer, n_embd 4, a 16-token
# vocabulary starting "<s>", "</s>", "a", "b", eos_token_id 1. All layer
# weights are zero, so the hidden state is the token embedding, and the
# embeddings and output.weight are chosen so that greedy decoding goes
#   "a" -> "b" -> "</s>"
# Checks:
#   1. prompt "a": the stream holds "b", then ends with reason "eos", 1 token;
#   2. prompt "b": no token at all, reason "eos";
#   3. plain-text mode prints "ab", without the "</s>" text;
#   4. the same model without tokenizer.ggml.eos_token_id still runs to --n.
# Exits non-zero on any mismatch.
set -euo pipefail
cd "$(dirname "$0")/../.."

CLI=${EOSLLM_CLI:-tools/eosllm-cli/eosllm-cli}
TMPDIR_=$(mktemp -d -t eosllm_eos_stop.XXXXXX)
trap 'rm -rf "$TMPDIR_"' EXIT

write_model() {  # write_model <path> <with-eos: 1|0>
python3 - "$1" "$2" << 'PY'
import struct, sys

U32, STRING, ARRAY = 4, 8, 9
GGML_F32 = 0
ALIGN = 32
n_embd, n_ff, n_ctx = 4, 4, 16
tokens = ['<s>', '</s>'] + [chr(c) for c in range(ord('a'), ord('a') + 14)]
n_vocab = len(tokens)
BOS, EOS, A, B = 0, 1, 2, 3

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
if sys.argv[2] == '1':
    kvs.append(('tokenizer.ggml.eos_token_id', U32, EOS))

def rows(n, set_rows):
    m = [[0.0] * n_embd for _ in range(n)]
    for r, v in set_rows.items():
        m[r] = v
    return [x for row in m for x in row]

e0, e1 = [1.0, 0, 0, 0], [0, 1.0, 0, 0]
ones = [1.0] * n_embd
zeros = [0.0] * (n_embd * n_embd)
tensors = [
    # [n_embd, n_vocab] in GGUF order: one row of n_embd per token.
    ('token_embd.weight', [n_embd, n_vocab], rows(n_vocab, {A: e0, B: e1})),
    ('output_norm.weight', [n_embd], ones),
    # "a" (e0) scores "b" highest; "b" (e1) scores "</s>" highest.
    ('output.weight', [n_embd, n_vocab], rows(n_vocab, {B: e0, EOS: e1})),
    ('blk.0.attn_norm.weight', [n_embd], ones),
    ('blk.0.attn_q.weight', [n_embd, n_embd], zeros),
    ('blk.0.attn_k.weight', [n_embd, n_embd], zeros),
    ('blk.0.attn_v.weight', [n_embd, n_embd], zeros),
    ('blk.0.attn_output.weight', [n_embd, n_embd], zeros),
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
}

fail() { echo "FAIL: $1"; echo "$2"; exit 1; }

MODEL="$TMPDIR_/eos.gguf"
NOEOS="$TMPDIR_/no-eos.gguf"
write_model "$MODEL" 1
write_model "$NOEOS" 0

echo "--- 1. prompt \"a\", --stream-jsonl ---"
out=$("$CLI" --model "$MODEL" --prompt "a" --n 5 --stream-jsonl)
echo "$out"
expect='{"t":"b","i":3}
{"done":true,"reason":"eos","n_tokens":1,"ms":0,"tok_per_s":0}'
[ "$out" = "$expect" ] || fail "expected \"b\", then reason \"eos\" after 1 token" "$out"

echo "--- 2. prompt \"b\", --stream-jsonl ---"
out=$("$CLI" --model "$MODEL" --prompt "b" --n 5 --stream-jsonl)
echo "$out"
[ "$out" = '{"done":true,"reason":"eos","n_tokens":0,"ms":0,"tok_per_s":0}' ] \
  || fail "expected reason \"eos\" with no token" "$out"

echo "--- 3. prompt \"a\", plain text ---"
out=$("$CLI" --model "$MODEL" --prompt "a" --n 5)
printf '%s\n' "$out"
[ "$out" = "ab" ] || fail "expected \"ab\" without the </s> text" "$out"

echo "--- 4. no tokenizer.ggml.eos_token_id: runs to --n ---"
out=$("$CLI" --model "$NOEOS" --prompt "a" --n 5 --stream-jsonl)
echo "$out"
last=$(printf '%s\n' "$out" | tail -n 1)
[ "$last" = '{"done":true,"reason":"max","n_tokens":5,"ms":0,"tok_per_s":0}' ] \
  || fail "expected reason \"max\" after 5 tokens" "$out"

echo "stop-at-eos: OK (4 checks)"
