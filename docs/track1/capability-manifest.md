# Track 1: capability-manifest format for scoped inference

**Status:** format specification (2026-10-13). Part of the
tightly-coupled AI track: a model may access the camera but not the
mic — and that scope is declared, signed, and enforced, not implied
by the app that happens to load it.

## The manifest

Every model artifact carries a manifest next to its pin. The manifest
is a small JSON document:

```json
{
  "model": "vision-detector@sha256:1f2e…",
  "capabilities": [
    { "name": "camera_frames", "access": "read" },
    { "name": "mic_frames", "access": "none" }
  ],
  "backends": ["ethos-u55", "cmsis-nn", "cpu"],
  "max_arena_kb": 256
}
```

Fields:

| Field | Meaning |
|---|---|
| `model` | The registry pin (`name@sha256:…`) of the artifact this manifest describes. A manifest never floats free of a pinned digest. |
| `capabilities` | The capability claims the model needs, each a `{name, access}` pair from the shared vocabulary below. Undeclared means denied. |
| `backends` | The accelerator-HAL backends the model may run on (delegate names from the eAI HAL design). A backend not listed must not execute the model. |
| `max_arena_kb` | Upper bound on the tensor arena the loader may grant. The loader sizes from this, never from the model's own say-so at run time. |

## One vocabulary, org-wide

Capability `name` and `access` values reuse the eApps AI-capability
registry (`eApps` `schemas/ai-capabilities.registry.json`) verbatim —
the same vocabulary apps declare in `data/apps.json`, so an app
manifest and a model manifest compose instead of translating:

| Capability name | What it covers |
|---|---|
| `camera_frames` | Frames from the device camera (vision input). |
| `mic_frames` | Audio frames from the device microphone. |
| `speaker_frames` | Audio rendered through the device speaker. |
| `inference.local` | On-device model inference (compute only). |
| `inference.cloud` | Off-device (cloud) model inference — data leaves the device. |
| `npu` | Direct access to the on-device neural accelerator. |
| `location` | Device location fixes. |
| `contacts` | Access to the device contacts / people graph. |

Access levels, exactly as in the registry: `none`, `read`, `write`,
`readwrite`. A name not in the registry is invalid — loaders reject
the manifest rather than ignoring the unknown claim (fail closed).

## Enforcement sketch

1. **Verify before parse.** The loader checks the artifact digest
   against the pin — the same check `ebuild model verify` performs
   on the registry entry — and verifies the manifest's signature
   under the model-signing envelope (the eos #162 envelope analogy:
   manifest and weights are signed together, so neither can be
   swapped alone).
2. **Intersect with host policy.** The effective grant is the
   intersection of the manifest's claims and the host's policy for
   the calling app. A claim the host denies is dropped; if the model
   declares the capability as required to run, loading refuses
   outright instead of running partially scoped.
3. **Hand the delegate only the grant.** The accelerator-HAL
   delegate receives the granted capability set at create time and
   checks tensor access against it at invoke time. The model never
   sees a capability it was not granted.

## Worked example: camera-only vision model

A person-detector that must never touch audio declares:

```json
{
  "model": "person-detector@sha256:9ab0…",
  "capabilities": [
    { "name": "camera_frames", "access": "read" },
    { "name": "mic_frames", "access": "none" },
    { "name": "inference.local", "access": "read" },
    { "name": "npu", "access": "read" }
  ],
  "backends": ["ethos-u55", "cmsis-nn", "cpu"],
  "max_arena_kb": 192
}
```

`mic_frames: none` is stated, not omitted: an explicit `none` is a
promise a reviewer (and an auditor) can check, and any later manifest
edit that widens it is a visible, signable change.

## Cross-references

- eApps `schemas/ai-capabilities.registry.json` — the vocabulary source of truth.
- eAI `docs/track1/accelerator-hal-delegate.md` — the delegate that receives the grant.
- ebuild `docs/quantize.md` — the model registry and `ebuild model verify`.
- [`inference-service.md`](inference-service.md) — how eosllm's service consumes manifests at load time.
