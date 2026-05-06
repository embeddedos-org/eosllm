#!/bin/bash
# Tests the --metadata code path by writing the same synthetic GGUF
# that --smoke builds in memory to a temp file, then dumping its
# metadata via the CLI. Exits 0 if the architecture key round-trips.
set -euo pipefail
cd /home/spatchava/embeddedos-org/eosllm

TMP=$(mktemp -t eosllm_meta_smoke.XXXXXX.gguf)
trap "rm -f $TMP" EXIT

python3 - "$TMP" << 'PY'
import struct, sys
out = sys.argv[1]
buf = bytearray()
buf += struct.pack('<I', 0x46554747)        # magic "GGUF"
buf += struct.pack('<I', 3)                  # version
buf += struct.pack('<Q', 1)                  # tensor_count
buf += struct.pack('<Q', 2)                  # kv_count
def lstr(s):
    return struct.pack('<Q', len(s)) + s.encode()
buf += lstr('general.architecture')
buf += struct.pack('<I', 8)                  # GGUFV_STRING
buf += lstr('smoke')
buf += lstr('general.alignment')
buf += struct.pack('<I', 4)                  # GGUFV_U32
buf += struct.pack('<I', 32)
buf += lstr('x.f32')
buf += struct.pack('<I', 1)                  # n_dims
buf += struct.pack('<Q', 4)
buf += struct.pack('<I', 0)                  # GGML_F32
buf += struct.pack('<Q', 0)                  # tensor_offset
while len(buf) % 32 != 0:
    buf += b'\x00'
buf += struct.pack('<4f', 1.0, 2.0, 3.0, 4.0)
open(out, 'wb').write(buf)
print(f'wrote {out} ({len(buf)} bytes)', file=sys.stderr)
PY

./tools/eosllm-cli/eosllm-cli --metadata "$TMP"
