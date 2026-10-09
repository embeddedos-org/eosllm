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
