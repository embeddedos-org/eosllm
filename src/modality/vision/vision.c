/*
 * src/modality/vision/vision.c — Phase 3 ViT / SigLIP encoder skeleton.
 *
 * The patch-embed → transformer encoder → image-feature pooler chain
 * lives here. Phase 1 ships a registration stub so that capability
 * bits are wired correctly; Phase 3 adds the real implementation.
 *
 * Anticipated tensor names (Llava family):
 *   v.position_embd.weight
 *   v.cls_embd.weight
 *   v.patch_embd.{weight,bias}
 *   v.blk.{i}.{attn_q,attn_k,attn_v,attn_o,ffn_up,ffn_down}.weight
 *   v.encoder_norm.weight
 *   v.mm_proj.{weight,bias}            (LLaVA cross-modal projector)
 *
 * Anticipated image input layout (per call to encode):
 *   const float *pixels   - RGB, [3, H, W], normalized to [-1,+1]
 */
#include <stddef.h>
#include "eosllm/eosllm.h"
#include "eosllm/modality.h"
#include "../modality_internal.h"

#if EOSLLM_HAVE_MODALITY_VISION

static eos_status_t open_  (eos_session_t *s, eos_model_t *m, void **st)
    { (void)s; (void)m; (void)st; return EOS_E_UNSUPPORTED; }
static eos_status_t encode_(void *st, const void *d, size_t n)
    { (void)st; (void)d; (void)n; return EOS_E_UNSUPPORTED; }
static void         close_ (void *st) { (void)st; }

static const eos_modality_vt_t k_vision = {
    "vision-vit", EOS_MODALITY_VISION, open_, encode_, close_
};

eos_status_t eosi_modality_vision_register(void) {
    return eos_modality_register(&k_vision);
}

#else
eos_status_t eosi_modality_vision_register(void) { return EOS_E_UNSUPPORTED; }
#endif
