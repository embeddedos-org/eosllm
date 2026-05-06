# tests/data/

This directory holds model files used by the golden / oracle test
suites. The files themselves are **not** stored in the repository (too
large; license-encumbered). Populate with:

```
./tools/fetch_test_models.sh             # all default models
./tools/fetch_test_models.sh tinyllama   # just one
```

## Files referenced by the test suite

| Filename                                             | Used by                                  | Source                                                                 |
|------------------------------------------------------|------------------------------------------|------------------------------------------------------------------------|
| `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`               | Session 1 golden tokenizer test;<br>Session 2 logits oracle;<br>libfuzzer GGUF corpus seed | https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF |
| `meta-llama-3-8b-instruct.Q4_K_M.gguf`               | Session 2 release-only logits oracle     | https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF (gated) |
| `bitnet-1.58b.gguf`                                  | Session 8 BitNet end-to-end              | https://huggingface.co/1bitLLM/bitnet_b1_58-large                      |
| `llava-1.5-7b.Q4_K_M.gguf` + `mmproj-llava-1.5-7b.gguf` | Session 13 multi-modal image caption  | https://huggingface.co/mys/ggml_llava-v1.5-7b                          |
| `whisper-tiny.bin`                                   | Session 13 audio transcription           | https://huggingface.co/ggerganov/whisper.cpp                           |

The fetch script verifies SHA-256 after download. To add a new file,
extend `tools/fetch_test_models.sh` with a new `(URL, sha256, dest)`
entry.

## Why these specific quantizations

- **TinyLlama Q4_K_M** is the smallest serious chat model (~700 MB) and
  the same quant the v0.1 acceptance criteria target. Fits in CI runner
  RAM; encodes/decodes the same vocab (Llama-2 BPE) as the larger
  Llama-2 family, so it doubles as a tokenizer regression target.
- **Llama-3-8B Q4_K_M** is the v0.1 "real model" parity target vs
  `llama.cpp`. Q4_K_M (not Q4_0) because that's what the upstream
  Bartowski quants ship as the recommended default.
- BitNet / Llava / Whisper details TBD as we approach those sessions.
