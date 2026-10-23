#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Conformer self-attention with relative positions on AMX-FP16 tiles (Granite Rapids and later), as a custom op:
// for query i and key j of head h,
//   out[h][i] = sum_j softmax_j(scale * (q_u[h][i] . k[h][j] + q_v[h][i] . p[h][T - 1 - i + j])) v[h][j]
// q_u, q_v, k, v [D, T, H, 1] and p [D, 2T - 1, H, 1] f32 with contiguous rows (any other strides); the result is
// [D, H, T, 1] f32, the layout of ggml_flash_attn_ext. It replaces the f16 casts of K and V, the position-score mask
// and flash attention. The queries, keys, values, position projections and softmax weights are rounded to f16 and
// the products summed in f32, so the results differ from that chain in the last bits.
// Custom op arguments: q_u, k, v, q_v, p; userdata: the bits of scale. NULL where AMX-FP16 is not built in or not
// usable, or for other shapes (D a multiple of 32, at most 512).
ggml_custom_op_t ggml_cpu_rel_pos_attn_amx_op(const struct ggml_tensor * q_u, const struct ggml_tensor * k,
                                              const struct ggml_tensor * v, const struct ggml_tensor * q_v,
                                              const struct ggml_tensor * p);

#ifdef __cplusplus
}
#endif
