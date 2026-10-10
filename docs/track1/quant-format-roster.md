# Quantization format roster (Track 1)

Answers the standing Track 1 open question: which quant formats does
the org produce, who consumes them, and what status is each at?
One table, kept current as formats move from experimental to default.

| Format | Producer | Consumer | Target class | Status |
|---|---|---|---|---|
| int8 (LiteRT-compatible) | `ebuild quantize --format int8` | eAI runtime, eosllm | MCU + application processors | **Default** — recorded in `ebuild/docs/quantize-mcu-target.md` |
| bitnet-1.58 (ternary, {-1,0,+1}) | `ebuild quantize --format bitnet-1.58` | eAI ternary tier | MCU (ESP32-S3-class, flash-resident) | Experimental — see `eAI/docs/track1/accelerator-hal-ternary-tier.md` |
| eAI ternary tier (weights in {-1,0,+1}, no FP MACs) | eAI accelerator HAL | eosllm inference service | MCU | Design — `eAI/docs/track1/accelerator-hal-ternary-tier.md` |
| `.eni_model` (real quantized payloads) | eNI tooling | eNI neural pipelines | MCU (on-device inference) | Live — fail-closed since eNI #40; see `eNI/docs/on-device-inference-reference.md` |

Notes:

- eosllm's own supported schemes are documented in
  [`docs/quant_schemes.md`](quant_schemes.md); this roster is the
  org-wide view and defers to that doc for eosllm-native detail.
- The fail-closed invariant crosses every row: a producer that cannot
  emit a real quantized payload must refuse (the eNI #40 precedent),
  never emit zero-weight placeholders.
- Cross-repo metric: **tokens-per-watt** (adopted for the accelerator
  HAL tiers) is how formats are compared once they run on hardware.

## SLM candidates for the ternary / BitNet-1.58 tier

Answers the standing 10-08 open question: which small models belong on the
ternary tier. Candidates are judged on (a) a real ternary or ternary-able
checkpoint, (b) MCU-plausible size, (c) a producer that can emit the
`bitnet-1.58` artifact above.

| Model | Size class | Ternary story | Producer -> consumer | Status |
|---|---|---|---|---|
| BitNet b1.58 2B4T-class | ~2B params | Native 1.58-bit weights ({-1,0,+1}) | `ebuild quantize --format bitnet-1.58` -> eAI ternary tier | Reference |
| TinyLlama 1.1B | 1.1B params | Ternary-able via post-training quantization; the federated-learning datapoint (`docs/track1/federated-learning-edge.md`) shows it fine-tunes on edge clients | `ebuild quantize` -> eAI runtime / eosllm | Candidate |
| SmolLM2 1.7B-class | ~1.7B params | Strong small-model baseline; ternary conversion experimental | `ebuild quantize --format bitnet-1.58` (experimental) -> eAI ternary tier | Candidate |

Notes:

- "Reference" means the tier was designed around it (the ESP32-S3
  0.4B-parameter SPI daisy-chain demo); "Candidate" means the conversion
  path exists but is not yet validated end-to-end on the tier.
- No model ships on the ternary tier until its converted artifact passes
  the tier's validation gate — the same fail-closed invariant as the
  format rows above.
