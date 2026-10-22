/*
 * include/transcribe/canary.h - Canary-family public extension surface.
 *
 * Includes transcribe.h; safe to include in C or C++ TUs. Holds the Canary
 * run extension: an optional draft transcript that the decoder checks
 * several tokens at a time. Probe via transcribe_model_accepts_ext_kind
 * before pointing transcribe_run_params::family at this struct.
 *
 * FourCC kinds are reserved in docs/extension-kinds.md.
 */

#ifndef TRANSCRIBE_CANARY_H
#define TRANSCRIBE_CANARY_H

#include "transcribe.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 'CNRN' little-endian = 0x4E524E43 */
#define TRANSCRIBE_EXT_KIND_CANARY_RUN 0x4E524E43u

/*
 * Canary run knobs. Reached via transcribe_run_params::family; NULL family
 * decodes as before.
 *
 *   draft
 *
 *     Expected transcript of the audio, for example the text of a first
 *     recognition pass. The decoder appends up to draft_max of its tokens to
 *     each decoder pass and keeps those that equal its own greedy choice, so
 *     the transcript is the same as without a draft; a draft close to the
 *     result only makes decoding faster. NULL or "" decodes one token per
 *     pass. The text is copied before transcribe_run returns. Used on the CPU
 *     backend for inputs under about 40 s, where the checked tokens take the
 *     same kernels as one-token steps; ignored elsewhere.
 *
 *   draft_max
 *
 *     Draft tokens checked per decoder pass, 1..32; <= 0 selects 8.
 */
struct transcribe_canary_run_ext {
    struct transcribe_ext ext;

    const char * draft;
    int32_t      draft_max;
};

/* Defaults: ext.size / ext.kind set, no draft. */
TRANSCRIBE_API void transcribe_canary_run_ext_init(struct transcribe_canary_run_ext * p);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TRANSCRIBE_CANARY_H */
