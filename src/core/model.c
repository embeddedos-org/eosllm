/*
 * src/core/model.c — public model accessors that dispatch to the
 * underlying format reader's ops vtable.
 */
#include <stddef.h>
#include <string.h>

#include "eosllm/model.h"
#include "eosllm/eosllm.h"

#include "internal.h"
#include "model_internal.h"

/*
 * eos_model_open walks the format registry, probes each, and asks
 * the first matching format to open the file via the OS shim's file
 * I/O. Implementation lives here so it doesn't need to be repeated
 * per format.
 */

/* Stream backed by the OS-shim file ops; defined below. */
static int64_t shim_read (void *opaque, void *buf, size_t len);
static int     shim_seek (void *opaque, int64_t offset);
static int64_t shim_size (void *opaque);
static void    shim_close(void *opaque);

eos_status_t eos_model_open(const char *path, eos_model_t **out_model) {
    eosi_registry_t *r = eosi_registry();
    eos_stream_t  s;
    void         *handle = NULL;
    size_t        i;
    eos_status_t  rc;

    /* Clear thread-local last-error so the slot reflects only this
     * call's outcome (cleared on EOS_OK, set by EOSI_LOG_ERROR sites
     * on failure paths). See docs/abi.md "Last-error string". */
    eosi_set_error(NULL);

    if (path == NULL || out_model == NULL) return EOS_E_INVALID_ARG;
    *out_model = NULL;
    if (!r->os_provided || r->os.file_open == NULL) return EOS_E_UNSUPPORTED;
    if (r->n_formats == 0) return EOS_E_UNSUPPORTED;

    if (r->os.file_open(path, &handle) != 0 || handle == NULL) {
        EOSI_LOG_ERROR("eos_model_open: os.file_open failed (file missing or unreadable)");
        return EOS_E_IO;
    }
    s.opaque = handle;
    s.read   = shim_read;
    s.seek   = shim_seek;
    s.size   = shim_size;
    s.close  = shim_close;

    for (i = 0; i < r->n_formats; ++i) {
        const eos_format_vt_t *fmt = r->formats[i];
        if (fmt->probe == NULL) continue;
        if (s.seek(handle, 0) != 0) { shim_close(handle); return EOS_E_IO; }
        rc = fmt->probe(&s);
        if (rc != EOS_OK) continue;
        if (s.seek(handle, 0) != 0) { shim_close(handle); return EOS_E_IO; }
        rc = fmt->open(&s, out_model);
        /* The format reader takes ownership of the stream on success;
         * on failure we still need to close it. */
        if (rc != EOS_OK) {
            shim_close(handle);
            return rc;
        }
        return EOS_OK;
    }

    shim_close(handle);
    return EOS_E_FORMAT;
}

void eos_model_close(eos_model_t *model) {
    if (model == NULL) return;
    if (model->ops != NULL && model->ops->close != NULL) {
        model->ops->close(model->impl);
    }
    eosi_free(model);
}

size_t eos_model_num_tensors(const eos_model_t *model) {
    eosi_set_error(NULL);
    if (model == NULL || model->ops == NULL) return 0;
    return model->ops->num_tensors(model->impl);
}

const eos_tensor_t *eos_model_tensor(const eos_model_t *model, size_t i) {
    eosi_set_error(NULL);
    if (model == NULL || model->ops == NULL) return NULL;
    return model->ops->tensor(model->impl, i);
}

const char *eos_model_meta_str(const eos_model_t *model, const char *key) {
    eosi_set_error(NULL);
    if (model == NULL || model->ops == NULL || key == NULL) return NULL;
    return model->ops->meta_str(model->impl, key);
}

eos_status_t eos_model_meta_u64(const eos_model_t *model, const char *key,
                                uint64_t *out_value) {
    eosi_set_error(NULL);
    if (model == NULL || model->ops == NULL || key == NULL || out_value == NULL)
        return EOS_E_INVALID_ARG;
    return model->ops->meta_u64(model->impl, key, out_value);
}

eos_status_t eos_model_meta_f32(const eos_model_t *model, const char *key,
                                float *out_value) {
    eosi_set_error(NULL);
    if (model == NULL || model->ops == NULL || key == NULL || out_value == NULL)
        return EOS_E_INVALID_ARG;
    if (model->ops->meta_f32 == NULL) return EOS_E_UNSUPPORTED;
    return model->ops->meta_f32(model->impl, key, out_value);
}

eos_status_t eos_model_meta_arr_str_count(const eos_model_t *model,
                                          const char *key, size_t *out_count) {
    eosi_set_error(NULL);
    if (model == NULL || model->ops == NULL || key == NULL || out_count == NULL)
        return EOS_E_INVALID_ARG;
    if (model->ops->meta_arr_str == NULL) return EOS_E_UNSUPPORTED;
    {
        size_t n = 0;
        eos_status_t s = model->ops->meta_arr_str(model->impl, key, NULL, &n);
        if (s == EOS_OK) *out_count = n;
        return s;
    }
}

eos_status_t eos_model_meta_arr_str_load(const eos_model_t *model,
                                         const char *key,
                                         const char **out_strs, size_t *io_n) {
    eosi_set_error(NULL);
    if (model == NULL || model->ops == NULL || key == NULL || io_n == NULL)
        return EOS_E_INVALID_ARG;
    if (model->ops->meta_arr_str == NULL) return EOS_E_UNSUPPORTED;
    return model->ops->meta_arr_str(model->impl, key, out_strs, io_n);
}

eos_status_t eos_model_meta_arr_u32_count(const eos_model_t *model,
                                          const char *key, size_t *out_count) {
    eosi_set_error(NULL);
    if (model == NULL || model->ops == NULL || key == NULL || out_count == NULL)
        return EOS_E_INVALID_ARG;
    if (model->ops->meta_arr_u32 == NULL) return EOS_E_UNSUPPORTED;
    {
        size_t n = 0;
        eos_status_t s = model->ops->meta_arr_u32(model->impl, key, NULL, &n);
        if (s == EOS_OK) *out_count = n;
        return s;
    }
}

eos_status_t eos_model_meta_arr_u32_load(const eos_model_t *model,
                                         const char *key,
                                         uint32_t *out_vals, size_t *io_n) {
    eosi_set_error(NULL);
    if (model == NULL || model->ops == NULL || key == NULL || io_n == NULL)
        return EOS_E_INVALID_ARG;
    if (model->ops->meta_arr_u32 == NULL) return EOS_E_UNSUPPORTED;
    return model->ops->meta_arr_u32(model->impl, key, out_vals, io_n);
}

const eos_tensor_t *eos_model_tensor_by_name(const eos_model_t *model,
                                             const char *name) {
    eosi_set_error(NULL);
    if (model == NULL || model->ops == NULL || name == NULL) return NULL;
    if (model->ops->tensor_by_name == NULL) return NULL;
    return model->ops->tensor_by_name(model->impl, name);
}

/* ------------------------------------------------------------------ */
/* Stream adapter                                                      */
/* ------------------------------------------------------------------ */

static int64_t shim_read(void *opaque, void *buf, size_t len) {
    eosi_registry_t *r = eosi_registry();
    return r->os.file_read(opaque, buf, len);
}

static int shim_seek(void *opaque, int64_t offset) {
    eosi_registry_t *r = eosi_registry();
    return r->os.file_seek(opaque, offset);
}

static int64_t shim_size(void *opaque) {
    eosi_registry_t *r = eosi_registry();
    return r->os.file_size(opaque);
}

static void shim_close(void *opaque) {
    eosi_registry_t *r = eosi_registry();
    if (r->os.file_close != NULL) r->os.file_close(opaque);
}
