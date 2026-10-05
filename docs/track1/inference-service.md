# eosllm as the inference-service implementation

**Status:** Design sketch (Track 1 — tightly-coupled AI).
**Date:** 2026-10-05
**Cross-links:** `eAI` `docs/track1/runtime-api.md` (the API surface),
`eos` `docs/track1/accelerator-hal-profiles.md` (backend tiers),
`eIPC` `docs/track1/zero-copy-tensor-ipc.md` (tensor transport).
**Vision issue:** [#16](https://github.com/embeddedos-org/eosllm/issues/16)
(UNIVERSAL AI CORE) — milestone proposal posted there; this doc is the M1
technical shape.

## The move: libraries → subsystem

Today eosllm is a set of libraries (loader, runtime, server, CLI). Track 1
makes inference a *subsystem*: eosllm becomes the first implementation of
the eAI inference-service API, running as an eos service rather than
something applications link directly.

## Module → API mapping

| Current eosllm module | Becomes |
|---|---|
| model loader | `eai_load_model` — plus envelope verification (gap, see below) |
| memory/allocator | `eai_alloc_arena` / `eai_free_arena` — the no-malloc hot-path contract (gap) |
| inference loop | `eai_infer` — capability argument plumbed from the app manifest (gap) |
| server / CLI | thin clients of the service; the CLI keeps its UX, loses its direct backend access |

## Gap list for M1

1. **Envelope verification in the loader.** Models must verify the Ed25519
   `.eapp` envelope (eos #162) before any weight byte is mapped. Currently
   the loader trusts the file it is given.
2. **Arena allocator.** Replace ad-hoc allocation with the sized-arena
   contract: the service sizes the arena from `scratch_bytes` +
   `quant_format`, the caller owns the memory, no heap on the hot path.
3. **Capability plumbing.** Thread `eai_capability_t` from the app manifest's
   `ai_capabilities` (eApps) through load (claim check) to infer (TrustZone
   attribution is configured by the platform layer; the service passes the
   claims down).
4. **Backend modules.** Current backends become `eos_accel_backend_t`
   modules (dispatch + scratch-sizing + quantization descriptor) so the
   service core stays vendor-neutral.

## What this is not

Not a rewrite of the inference math — the kernels stay. Not the UNIVERSAL
AI CORE orchestrator itself (#16 M4); this doc is the *substrate* milestones
M1–M3 build on: a trustworthy, capability-scoped, arena-bounded service the
orchestrator can call without becoming the trust boundary.
