/*
 * src/core/caps.c — eos_caps() reports what was compiled in.
 *
 * Each EOSLLM_HAVE_* is guaranteed to be defined as 0 or 1 by the
 * Makefile, so a plain assignment works.
 */
#include <string.h>

#include "eosllm/eosllm.h"
#include "internal.h"

eos_status_t eos_caps(eos_caps_t *out) {
    if (out == NULL) return EOS_E_INVALID_ARG;

    memset(out, 0, sizeof(*out));

    out->have_posix      = (EOSLLM_HAVE_POSIX        != 0);
    out->have_zephyr     = (EOSLLM_HAVE_ZEPHYR       != 0);
    out->have_freertos   = (EOSLLM_HAVE_FREERTOS     != 0);
    out->have_baremetal  = (EOSLLM_HAVE_BAREMETAL    != 0);

    out->have_threads    = (EOSLLM_HAVE_THREADS      != 0);

    out->have_k_scalar   = (EOSLLM_HAVE_KERNEL_SCALAR != 0);
    out->have_k_avx2     = (EOSLLM_HAVE_KERNEL_AVX2   != 0);
    out->have_k_avx512   = (EOSLLM_HAVE_KERNEL_AVX512 != 0);
    out->have_k_neon     = (EOSLLM_HAVE_KERNEL_NEON   != 0);
    out->have_k_sve      = (EOSLLM_HAVE_KERNEL_SVE    != 0);
    out->have_k_rvv      = (EOSLLM_HAVE_KERNEL_RVV    != 0);
    out->have_k_hvx      = (EOSLLM_HAVE_KERNEL_HVX    != 0);
    out->have_k_npu      = (EOSLLM_HAVE_KERNEL_NPU    != 0);

    out->have_q_q8_0     = (EOSLLM_HAVE_QUANT_Q8_0       != 0);
    out->have_q_q4_k     = (EOSLLM_HAVE_QUANT_Q4_K       != 0);
    out->have_q_q2_k     = (EOSLLM_HAVE_QUANT_Q2_K       != 0);
    out->have_q_q1_58    = (EOSLLM_HAVE_QUANT_Q1_58      != 0);
    out->have_q_mixed    = (EOSLLM_HAVE_QUANT_MIXED      != 0);
    out->have_q_calibrtd = (EOSLLM_HAVE_QUANT_CALIBRATED != 0);

    out->have_m_text     = (EOSLLM_HAVE_MODALITY_TEXT   != 0);
    out->have_m_vision   = (EOSLLM_HAVE_MODALITY_VISION != 0);
    out->have_m_audio    = (EOSLLM_HAVE_MODALITY_AUDIO  != 0);

    out->have_t_bpe      = (EOSLLM_HAVE_TOKENIZER_BPE != 0);
    out->have_t_spm      = (EOSLLM_HAVE_TOKENIZER_SPM != 0);

    out->have_f_eosm     = (EOSLLM_HAVE_FORMAT_EOSM != 0);
    out->have_f_gguf     = (EOSLLM_HAVE_FORMAT_GGUF != 0);

    out->have_s_greedy   = (EOSLLM_HAVE_SCHED_GREEDY   != 0);
    out->have_s_deadline = (EOSLLM_HAVE_SCHED_DEADLINE != 0);
    out->have_s_batched  = (EOSLLM_HAVE_SCHED_BATCHED  != 0);

    /* No on-disk format implementation exists in Phase 0. */
    out->eosm_max_version = 0;

    return EOS_OK;
}
