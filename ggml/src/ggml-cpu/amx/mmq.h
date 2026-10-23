#pragma once
#include "common.h"

size_t ggml_backend_amx_desired_wsize(const struct ggml_tensor * dst);

size_t ggml_backend_amx_get_alloc_size(const struct ggml_tensor * tensor);

// bytes of a Q8_0 weight packed in the standard layout
size_t amx_q8_0_packed_size(int64_t N, int64_t K);

void ggml_backend_amx_convert_weight(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);

void ggml_backend_amx_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst);

// MUL_MAT + ADD of a bias row in one pass, into add's data; false if this shape is not handled
bool ggml_backend_amx_mul_mat_add(const struct ggml_compute_params * params, struct ggml_tensor * mm, struct ggml_tensor * add);
