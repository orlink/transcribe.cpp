#include "ggml-backend.h"
#include "ggml-cpu-impl.h"

// GGML internal header

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
ggml_backend_buffer_type_t ggml_backend_amx_buffer_type(void);

// Opt-in for the grouped Q8_0 layout (amx/mmq-grouped.h) of weights converted into the AMX buffer from this
// thread: act_group < 0 off (the default), 0 one activation scale per row, > 0 values per activation scale.
// A weight takes the grouped layout only if its blocks share one scale per group of values.
void ggml_backend_amx_set_q8_0_grouped(int act_group);
#endif
