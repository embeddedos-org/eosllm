#!/bin/bash
# `eosllm-cli --metadata` must print general.file_type.
#
# GGUF stores general.file_type as a uint32 (the llama.cpp writers and
# tools/eosllm-convert both use U32), so it has to be read with
# eos_model_meta_u64(), not eos_model_meta_str().
#
# Writes a tiny GGUF with general.architecture, general.name and
# general.file_type = 15, runs --metadata on it and checks that the output
# is JSON and holds all three keys with the right values.
# Exits non-zero on any mismatch.
set -euo pipefail
cd "$(dirname "$0")/../.."

CLI=${EOSLLM_CLI:-tools/eosllm-cli/eosllm-cli}
TMPDIR_=$(mktemp -d -t eosllm_meta_ftype.XXXXXX)
trap 'rm -rf "$TMPDIR_"' EXIT

MODEL="$TMPDIR_/ftype.gguf"
python3 - "$MODEL" << 'PY'
import struct, sys

U32, STRING = 4, 8
GGML_F32 = 0
ALIGN = 32

def lstr(s):
    b = s.encode()
    return struct.pack('<Q', len(b)) + b

kvs = [
    ('general.architecture', STRING, 'llama'),
    ('general.name', STRING, 'ftype-test'),
    ('general.alignment', U32, ALIGN),
    ('general.file_type', U32, 15),
]

buf = bytearray()
buf += struct.pack('<I', 0x46554747)   # magic "GGUF"
buf += struct.pack('<I', 3)            # version
buf += struct.pack('<Q', 1)            # tensor_count
buf += struct.pack('<Q', len(kvs))     # kv_count
for key, typ, val in kvs:
    buf += lstr(key) + struct.pack('<I', typ)
    buf += lstr(val) if typ == STRING else struct.pack('<I', val)
buf += lstr('x.f32')
buf += struct.pack('<I', 1)            # n_dims
buf += struct.pack('<Q', 4)
buf += struct.pack('<I', GGML_F32)
buf += struct.pack('<Q', 0)            # tensor offset
while len(buf) % ALIGN:
    buf += b'\x00'
buf += struct.pack('<4f', 1.0, 2.0, 3.0, 4.0)
open(sys.argv[1], 'wb').write(buf)
PY

OUT="$TMPDIR_/meta.json"
"$CLI" --metadata "$MODEL" > "$OUT"
cat "$OUT"

python3 - "$OUT" << 'PY'
import json, sys
meta = json.load(open(sys.argv[1]))["metadata"]
want = {
    "general.architecture": "llama",
    "general.name": "ftype-test",
    "general.file_type": 15,
}
bad = [k for k, v in want.items() if meta.get(k) != v]
if bad:
    for k in bad:
        print(f"FAIL: {k}: expected {want[k]!r}, got {meta.get(k, '<missing>')!r}")
    sys.exit(1)
print("PASS: --metadata prints general.file_type = 15")
PY
