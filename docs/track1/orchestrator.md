# The Universal AI Core, scoped (Track 1)

**Status:** Design (Track 1 — tightly-coupled AI).
**Date:** 2026-10-11
**Cross-links:** [`inference-service.md`](inference-service.md) (the M1
substrate the orchestrator calls),
[`quant-format-roster.md`](quant-format-roster.md) (which formats and
model classes the registry can hold), `ebuild` (the registry producer),
`eos` (the #162 envelope the capability model borrows), `eApps` (the app
manifest whose `ai_capabilities` the orchestrator inherits).
**Vision issue:** [#16](https://github.com/embeddedos-org/eosllm/issues/16)
(UNIVERSAL AI CORE) — this doc scopes it; the issue is closed by it.

## The scoping decision

Issue #16 proposed a "Universal Local AI": one UI backed by an
orchestrator LLM that understands tasks, chooses specialist models from a
registry, combines their results, and remembers them — the "it feels like I
own all models" experience. That vision is worth keeping.

What is **not** being kept is the open-ended 100–200 GB model library that
the issue anchored on. A device cannot hold hundreds of gigabytes of
models, swap them all, and still meet a power budget; and the "absorb every
model's weights into one brain" reading is not physically possible across
unrelated architectures. Track 1 instead treats AI as an *OS subsystem*
(see `inference-service.md`): the orchestrator is a policy layer on top of
a small, curated, capability-scoped set of signed models, all on-device.

## Architecture

```text
              ┌─────────────────────┐
              │      ONE AI / UI    │  the user sees one assistant
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │  ORCHESTRATOR LLM   │  understands intent, plans the call
              │                     │  graph, synthesizes the answer
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │   MODEL REGISTRY    │  .ebuild/models.json — what is
              │                     │  installed, its capabilities, and
              │                     │  its signed envelope
              └──────────┬──────────┘
                         ▼
        ┌────────────────┴────────────────┐
        ▼                ▼                ▼
   general         vision/STT/etc.    specialist
   (fast mode: orchestrator answers alone; routing mode: one specialist;
    fusion mode: several specialists + synthesis)
        │                │                │
        └────────────────┴────────────────┘
                         ▼
              ┌─────────────────────┐
              │  eAI inference      │  the ONLY way models execute:
              │  service            │  envelope-verified, arena-bounded,
              └─────────────────────┘  capability-checked (M1–M3)
```

The orchestrator never runs inference itself and never becomes the trust
boundary. Every model call goes through the M1–M3 inference service, so
models the orchestrator delegates to are still envelope-verified
(Eduardo-signed `.eapp` v2, eos #162), arena-bounded (no-malloc hot path),
and capability-checked. The #16 "three modes" survive as policies on this
stack: fast (no delegation), routing (one specialist), fusion (several,
with the orchestrator synthesizing).

## Model registry: tied to ebuild

The registry is not a hand-maintained list. `ebuild quantize` — the step
that converts and quantizes a model for a target class — is the registry
producer: each successful quantize run appends (or refreshes) one entry in
the project-local `.ebuild/models.json`. The orchestrator reads this file;
nothing else is allowed to register a model, so the registry always
describes artifacts ebuild actually produced for the hardware at hand.

### Registry schema (v1)

```json
{
  "registry_version": 1,
  "device": "nxp-ara240-evk",
  "models": [
    {
      "id": "orchestrator-q4_0",
      "kind": "orchestrator",
      "task": ["chat", "plan", "synthesize"],
      "artifact": ".ebuild/models/orchestrator-q4_0.eapp",
      "format": "int4",
      "params_b": 3.8,
      "size_mb": 2100,
      "ram_mb": 2600,
      "envelope": {
        "signature": "ed25519:...",
        "capabilities": ["ai.orchestrate", "camera.read"]
      },
      "capability_scope": ["ai.orchestrate", "camera.read"],
      "ebuild_run": "quantize-2026-10-11T09-00-00Z"
    },
    {
      "id": "vision-s1_58",
      "kind": "specialist",
      "task": ["vision"],
      "artifact": ".ebuild/models/vision-s1_58.eapp",
      "format": "bitnet-1.58",
      "params_b": 2.0,
      "size_mb": 260,
      "ram_mb": 400,
      "envelope": {
        "signature": "ed25519:...",
        "capabilities": ["camera.read"]
      },
      "capability_scope": ["camera.read"],
      "ebuild_run": "quantize-2026-10-11T09-04-00Z"
    }
  ]
}
```

Required fields: `id`, `kind` (`orchestrator` | `specialist`), `task`,
`artifact`, `format`, `size_mb`, `envelope.signature`,
`capability_scope`. Optional but recommended: `params_b`, `ram_mb`,
`ebuild_run`. The orchestrator rejects any entry whose artifact fails
envelope verification or whose `capability_scope` is absent.

## Capability-scoped inference

A model may access the camera but not the mic — that is the unit of
scoping. Capabilities are declared in the registry entry's
`capability_scope` and carried in the model's `.eapp` envelope, exactly
like eos's #162 package-signature envelope: the loader verifies the Ed25519
signature **and** the declared capability set before any weight byte is
mapped, and the inference service (`eai_infer`) receives the resolved
capability set per call.

Capability declaration format: dot-namespaced strings in three families:

- `ai.*` — orchestrator powers: `ai.orchestrate` (delegate to other
  models), `ai.speak` (text-to-speech), `ai.listen` (speech-to-text).
- `<sensor>.*` — physical inputs: `camera.read`, `mic.read`, `gnss.read`,
  `imu.read`. A vision model holds `camera.read` and nothing else.
- `<service>.*` — OS services: `net.fetch` (model may trigger a fetch),
  `storage.read`, `storage.write`.

Rules:

1. A model's *effective* capabilities are the intersection of the
   registry's `capability_scope`, the envelope's declared capabilities,
   and the calling app's `ai_capabilities` manifest (eApps). Narrowest
   wins; a mismatch is a fail-closed error, not a downgrade.
2. The orchestrator may only delegate to a specialist whose
   `capability_scope` covers what the delegated subtask needs. It cannot
   lend a specialist a capability the specialist does not itself hold.
3. Capability grants are per-session, logged, and revocable; `ai.listen`
   and `mic.read` additionally require the platform's user-consent gate.

## What this is not

- **Not a 100–200 GB model store.** The registry holds what ebuild
  produced for *this* device — typically a handful of quantized models
  totaling low gigabytes. The issue's model-budget sketch is superseded
  by per-device budgets sized from `ram_mb` and the target class.
- **Not a cloud dependency.** All inference is on-device. `net.fetch` is
  a capability a model may hold, not a requirement; the orchestrator
  functions fully offline.
- **Not weight fusion.** The issue's "absorb all models into one brain"
  reading is out of scope. Fusion happens at the *output* level —
  orchestrator synthesis over specialist results — which is the only
  mode that works across unrelated architectures.
- **Not the M1–M3 substrate.** Loading, arenas, and envelope mechanics
  live in `inference-service.md`; this doc is the policy layer on top.

## Hardware target: NXP Ara240

The concrete "runs large models at the edge" target is the
**NXP Ara240**: an application-processor-class edge SoC with a neural
processing unit. The orchestrator design is validated against it first —
registry budgets, `ram_mb` sizing, and capability enforcement all ship with
an Ara240 profile before any other target is added. MCU-class targets
(ESP32-S3-class, ternary tier) stay valid for specialists only; the
orchestrator itself is an application-processor workload.

## Acceptance criteria

1. **Registry schema** — `.ebuild/models.json` schema v1 implemented as
   a JSON Schema, consumed by the orchestrator and produced by
   `ebuild quantize`; ebuild CI round-trips one quantize run into a
   valid registry entry.
2. **Capability declaration format** — the three families (`ai.*`,
   `<sensor>.*`, `<service>.*`) enforced: the loader refuses an entry
   with a malformed or empty `capability_scope`, and a unit test proves
   a `camera.read`-only model cannot reach the mic path (narrowest-wins
   intersection, fail-closed on mismatch).
3. **NXP Ara240** — the orchestrator + registry + capability checks run
   on the Ara240 profile: fast, routing, and fusion modes demonstrated
   with the orchestrator, a vision specialist, and a general specialist,
   all within the profile's RAM budget, fully offline.
4. **Closed loop** — issue #16 is closed by this doc; no new work is
   filed against the old 100–200 GB vision.
