#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Direct kernels, run as custom ops, for chains of ops that the CPU backend otherwise computes as several nodes
// (im2col and a matmul with separate bias and activation nodes, copies for transposes). Each kernel computes every
// output with the operations of that generic path, in the same order, so the results are bit-identical with it.
// A getter returns NULL where the generic path for the given tensors is not the one a kernel reproduces (other
// compilers, instruction sets or shapes); the caller then builds the generic ops. Unless noted otherwise, the
// custom op's userdata is the padding.

// x [W, H, 1, 1] f32 convolved with w [3, 3, 1, C] (f32 or f16), stride 2, the same padding on both axes, plus
// b [C] f32, then ReLU: [OW, OH, C, 1]. Reproduces ggml_conv_2d (f16 im2col and matmul) + ggml_add + ggml_relu.
ggml_custom_op_t ggml_cpu_conv_2d_s2_c1_bias_relu_op(const struct ggml_tensor * x, const struct ggml_tensor * w,
                                                     const struct ggml_tensor * b, int pad);

// Depthwise: x [W, H, C, 1] f32, w [3, 3, 1, C] f32, stride 2, the same padding on both axes, plus b [C] f32:
// [OW, OH, C, 1]. Reproduces an f32 im2col with a [9 x 1] matmul per channel (conformer conv_2d_dw_f32) +
// ggml_add.
ggml_custom_op_t ggml_cpu_conv_2d_dw_s2_bias_op(const struct ggml_tensor * x, const struct ggml_tensor * w,
                                                const struct ggml_tensor * b, int pad);

// Pointwise: x [W, H, Cin, 1] f32 with w [1, 1, Cin, Cout] (f32 or f16), plus b [Cout] f32, then ReLU:
// [W, H, Cout, 1]; pad must be 0. Reproduces ggml_conv_2d (f16 im2col and matmul) + ggml_add + ggml_relu.
ggml_custom_op_t ggml_cpu_conv_2d_pw_bias_relu_op(const struct ggml_tensor * x, const struct ggml_tensor * w,
                                                  const struct ggml_tensor * b, int pad);

// A conformer conv module after its gate: x [C, T, 1, 1] f32 (channels innermost), a depthwise conv along T with
// w [9, 1, C] f32 and padding 4, plus b [C] f32 (may be NULL), times scale [C], plus shift [C] (batch norm),
// then SiLU: [C, T, 1, 1]. Reproduces the transpose to [T, C], ggml_conv_2d_dw_direct, ggml_add, ggml_mul,
// ggml_add, ggml_silu and the transpose back. Custom op arguments: x, w, scale, shift, and b if not NULL.
ggml_custom_op_t ggml_cpu_conv_1d_dw_bn_silu_op(const struct ggml_tensor * x, const struct ggml_tensor * w,
                                                const struct ggml_tensor * b, const struct ggml_tensor * scale,
                                                const struct ggml_tensor * shift, int pad);

// Relative-position scores of conformer attention as the flash-attention mask: q [D, T, H, 1] and p [D, 2T - 1, H, 1]
// f32 (rows contiguous, any other strides), mask[h][i][j] = f16(scale * dot(q[h][i], p[h][T - 1 - i + j])):
// [T, T, H, 1] f16. Reproduces ggml_mul_mat(p, q) (one ggml_vec_dot_f32 per score), the relative shift and its
// copy, ggml_scale and the cast to f16, computing only the T x T scores kept of the 2T - 1 x T. The custom op's
// userdata holds the bits of scale.
ggml_custom_op_t ggml_cpu_rel_pos_scores_f16_op(const struct ggml_tensor * q, const struct ggml_tensor * p);

#ifdef __cplusplus
}
#endif
