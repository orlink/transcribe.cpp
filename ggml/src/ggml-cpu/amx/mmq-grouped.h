#pragma once
#include "common.h"

// AMX int8 matmul for Q8_0 weights whose blocks share one scale per group of values (opt-in at load time,
// see ggml_backend_amx_set_q8_0_grouped in amx.h).
//
// A Q8_0 tensor converted into the AMX buffer while the opt-in is on, and whose blocks share their scale
// within every group of G values of a row (G = the whole row, 512, 256, 128 or 64; the largest that holds),
// is stored in this layout instead of the Q8_0 tiles:
//   int8 tiles [N / 16][K / 64][16 x 64] (VNNI order: byte ((k % 64) / 4) * 64 + (n % 16) * 4 + k % 4),
//   then the weight scales as float [N / 16][K / G][16],
//   then 128 x the column sums of each scaling group (see below) as int32 [N / 16][K / min(G, A)][16].
// Activations are quantized to int8 per row with one scale per group of A values (A = the whole row, or a
// group size given with the opt-in), and the int32 sums are scaled once per min(G, A) values instead of once
// per 32: with AMX tiles of 16 x 16 x 64 for more than 8 rows, with AVX512-VNNI (and column sums made at
// load) for up to 8. A trailer after the standard packed size records the layout; every other tensor gets
// an empty trailer and keeps the standard kernels.

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)

struct amx_q8_0_grouped_info {
    uint64_t magic;
    int32_t  K, N;
    int32_t  wg;  // values per weight scale
    int32_t  ag;  // values per activation scale
};

// bytes after the standard packed Q8_0 data that every Q8_0 tensor in the AMX buffer gets
constexpr size_t AMX_Q8_0_TRAILER_SIZE = 64;

// set the trailer of a tensor packed the standard way
void amx_q8_0_grouped_clear(void * trailer);

// pack `data` (Q8_0 blocks of tensor t) in the grouped layout at t->data if the opt-in is on and the blocks
// share their scales per group; writes the trailer and returns true, or returns false (nothing written)
bool amx_q8_0_grouped_pack(const struct ggml_tensor * t, const void * data, void * trailer);

// the layout of a weight in the AMX buffer, or nullptr if it is not grouped; trailer: its trailer
const amx_q8_0_grouped_info * amx_q8_0_grouped_get(const struct ggml_tensor * src0, const void * trailer);

// work space of a grouped mul_mat with M activation rows
size_t amx_q8_0_grouped_wsize(const amx_q8_0_grouped_info & gi, int64_t M);

// dst (or out_data, with the bias row added) = src1 @ src0^T for a grouped weight
void amx_q8_0_grouped_mul_mat(const struct ggml_compute_params * params, const amx_q8_0_grouped_info & gi,
                              const struct ggml_tensor * src0, const struct ggml_tensor * src1,
                              struct ggml_tensor * dst, float * out_data, const float * bias);

#endif
