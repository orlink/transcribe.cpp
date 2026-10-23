// Direct kernels that reproduce the generic CPU path bit for bit (see fused-ops.h).

#include "fused-ops.h"

#include "ggml-cpu-impl.h"
#include "ops.h"
#include "simd-mappings.h"
#include "vec.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

// The kernels follow instruction sequences of GCC's AVX-512 build. Other builds keep the generic path.
#if defined(__AVX512F__) && defined(__GNUC__) && !defined(__clang__)
#define GGML_CONV_FUSED_AVX512 1
// conv kernels whose generic matmul is tinyBLAS (llamafile/sgemm.cpp), with the summation tree of GCC's
// _mm512_reduce_add_ps
#if defined(GGML_USE_LLAMAFILE)
#define GGML_CONV_FUSED_TINYBLAS 1
#endif
#endif

namespace {

#if defined(GGML_CONV_FUSED_AVX512)

// Plain products that GCC must not fuse with a following add (it fuses _mm512_mul_ps + _mm512_add_ps under
// -ffp-contract=fast, but not the _round_ forms).
inline __m512 mul_unfused(__m512 a, __m512 b) {
    return _mm512_mul_round_ps(a, b, _MM_FROUND_CUR_DIRECTION);
}

constexpr int64_t k_max_width    = 2048;
constexpr int64_t k_max_channels = 1024;

inline int64_t conv_out_size_k3s2(int64_t n, int pad) {
    return (n + 2 * pad - 3) / 2 + 1;
}

bool is_f32_vector(const ggml_tensor * t, int64_t n) {
    return t != nullptr && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_nelements(t) == n;
}

// One zero-padded input row split into even and odd columns, so that output column ow of a 3-wide stride-2
// window reads even[ow], odd[ow], even[ow + 1]. Filled up to n_vec 16-wide vectors and one more vector. buf is
// scratch of at least 32 (n_vec + 2) floats, zero outside [pad, pad + W) (it is only written there).
void split_row_even_odd(const float * row, int64_t W, int pad, bool valid, int64_t n_vec, float * buf, float * even,
                        float * odd) {
    if (!valid) {
        memset(even, 0, 16 * (n_vec + 1) * sizeof(float));
        memset(odd, 0, 16 * (n_vec + 1) * sizeof(float));
        return;
    }
    memcpy(buf + pad, row, W * sizeof(float));
    const __m512i ie = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    const __m512i io = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
    for (int64_t v = 0; v <= n_vec; ++v) {
        const __m512 a = _mm512_loadu_ps(buf + 32 * v);
        const __m512 b = _mm512_loadu_ps(buf + 32 * v + 16);
        _mm512_storeu_ps(even + 16 * v, _mm512_permutex2var_ps(a, ie, b));
        _mm512_storeu_ps(odd + 16 * v, _mm512_permutex2var_ps(a, io, b));
    }
}

#endif  // GGML_CONV_FUSED_AVX512

#if defined(GGML_CONV_FUSED_TINYBLAS)

inline float round_to_f16(float v) {
    return GGML_CPU_FP16_TO_FP32(GGML_CPU_FP32_TO_FP16(v));
}

// The generic path: im2col to f16 ([OW * OH, 9] rows), the kernel converted to f16, and tinyBLAS's AVX-512
// gemm with k = 9 < 16: one masked fmadd of the nine f16 products into zeros (exact, f16 products fit in f32),
// then _mm512_reduce_add_ps, whose tree over lanes 0-8 (lanes 9-15 zero) is
//   ((p4 + (p8 + p0)) + (p6 + p2)) + ((p5 + p1) + (p7 + p3)),
// then a separate bias add and ReLU. Here the lanes are 16 neighbouring outputs of one row and channel.
void conv_2d_s2_c1_bias_relu(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const ggml_tensor * x   = dst->src[0];
    const ggml_tensor * w   = dst->src[1];
    const ggml_tensor * b   = dst->src[2];
    const int           pad = (int) (intptr_t) userdata;

    const int64_t W  = x->ne[0];
    const int64_t H  = x->ne[1];
    const int64_t OW = dst->ne[0];
    const int64_t OH = dst->ne[1];
    const int64_t C  = dst->ne[2];

    const float * xd = (const float *) x->data;
    const float * bd = (const float *) b->data;
    float *       yd = (float *) dst->data;

    const int64_t oh0 = OH * ith / nth;
    const int64_t oh1 = OH * (ith + 1) / nth;
    if (oh0 >= oh1) {
        return;
    }

    // the kernel rounded to f16, as the matmul converts it
    float wk[9 * k_max_channels];
    for (int64_t i = 0; i < 9 * C; ++i) {
        wk[i] = w->type == GGML_TYPE_F16 ? GGML_CPU_FP16_TO_FP32(((const ggml_fp16_t *) w->data)[i]) :
                                           round_to_f16(((const float *) w->data)[i]);
    }

    // per kernel row, the f16-rounded zero-padded input row split into even and odd columns: output column ow
    // reads padded columns 2 ow, 2 ow + 1, 2 ow + 2 = even[ow], odd[ow], even[ow + 1]
    constexpr int64_t k_row = (k_max_width + 2 * 2) / 2 + 32;
    float             even[3][k_row];
    float             odd[3][k_row];
    float             buf[2 * k_row + 64] = {};
    const int64_t     n_vec = (OW + 15) / 16;

    const __m512 zero = _mm512_setzero_ps();

    for (int64_t oh = oh0; oh < oh1; ++oh) {
        for (int kh = 0; kh < 3; ++kh) {
            const int64_t ih = 2 * oh - pad + kh;
            const bool    in = ih >= 0 && ih < H;
            split_row_even_odd(in ? xd + ih * W : xd, W, pad, in, n_vec, buf, even[kh], odd[kh]);
            for (int64_t j = 0; j <= 16 * n_vec; ++j) {
                even[kh][j] = round_to_f16(even[kh][j]);
                odd[kh][j]  = round_to_f16(odd[kh][j]);
            }
        }
        for (int64_t c = 0; c < C; ++c) {
            __m512 wv[9];
            for (int k = 0; k < 9; ++k) {
                wv[k] = _mm512_set1_ps(wk[c * 9 + k]);
            }
            const __m512 bc  = _mm512_set1_ps(bd[c]);
            float *      out = yd + (c * OH + oh) * OW;
            for (int64_t v = 0; v < n_vec; ++v) {
                const int64_t ow0 = 16 * v;
                __m512        p[9];
                for (int kh = 0; kh < 3; ++kh) {
                    p[3 * kh + 0] = _mm512_fmadd_ps(_mm512_loadu_ps(even[kh] + ow0), wv[3 * kh + 0], zero);
                    p[3 * kh + 1] = _mm512_fmadd_ps(_mm512_loadu_ps(odd[kh] + ow0), wv[3 * kh + 1], zero);
                    p[3 * kh + 2] = _mm512_fmadd_ps(_mm512_loadu_ps(even[kh] + ow0 + 1), wv[3 * kh + 2], zero);
                }
                __m512 r = _mm512_add_ps(
                    _mm512_add_ps(_mm512_add_ps(p[4], _mm512_add_ps(p[8], p[0])), _mm512_add_ps(p[6], p[2])),
                    _mm512_add_ps(_mm512_add_ps(p[5], p[1]), _mm512_add_ps(p[7], p[3])));
                r = _mm512_max_ps(_mm512_add_ps(r, bc), zero);  // max(nan, 0) and max(-0, 0) give 0, as ReLU
                if (ow0 + 16 <= OW) {
                    _mm512_storeu_ps(out + ow0, r);
                } else {
                    _mm512_mask_storeu_ps(out + ow0, (__mmask16) ((1u << (OW - ow0)) - 1), r);
                }
            }
        }
    }
}

// The generic path of a 1x1 conv: im2col to f16 (each position's Cin values), the kernel converted to f16, and
// tinyBLAS's AVX-512 gemm with k = Cin a multiple of 16: per output, one vector accumulator whose lane m takes
// the f16 products of channels m, m + 16, m + 32, ... by FMA in that order, then _mm512_reduce_add_ps of the 16
// lanes, then a separate bias add and ReLU. Here the lanes are 16 neighbouring positions instead, with 16
// accumulators (one per lane m of the original) reduced by the same tree:
//   T3[i] = a[8 + i] + a[i], T6[i] = T3[4 + i] + T3[i], r = (T6[0] + T6[2]) + (T6[1] + T6[3]).
void conv_2d_pw_bias_relu(ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) userdata;
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * w = dst->src[1];
    const ggml_tensor * b = dst->src[2];

    const int64_t P    = x->ne[0] * x->ne[1];
    const int64_t Cin  = x->ne[2];
    const int64_t Cout = dst->ne[2];

    const float * xd = (const float *) x->data;
    const float * bd = (const float *) b->data;
    float *       yd = (float *) dst->data;

    // the kernel rounded to f16, [Cout][Cin]
    static thread_local std::vector<float> wk;
    wk.resize(Cin * Cout);
    for (int64_t i = 0; i < Cin * Cout; i += 16) {  // Cin is a multiple of 16
        const __m256i h = w->type == GGML_TYPE_F16 ?
                              _mm256_loadu_si256((const __m256i *) ((const ggml_fp16_t *) w->data + i)) :
                              _mm512_cvtps_ph(_mm512_loadu_ps((const float *) w->data + i),
                                              _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm512_storeu_ps(wk.data() + i, _mm512_cvtph_ps(h));
    }

    // positions in chunks of 16-wide blocks, split over the threads; the chunk's inputs rounded to f16, each
    // block's [Cin][16] values contiguous (so the Cin rows a block reads do not share a few cache sets)
    constexpr int64_t                      k_chunk = 256;
    static thread_local std::vector<float> xs;
    xs.resize(Cin * k_chunk);
    const int64_t n_blk = (P + 15) / 16;
    const int64_t blk0  = n_blk * ith / nth;
    const int64_t blk1  = n_blk * (ith + 1) / nth;

    for (int64_t cb = blk0; cb < blk1; cb += k_chunk / 16) {
        const int64_t p0 = 16 * cb;
        const int64_t np = std::min<int64_t>(16 * std::min<int64_t>(blk1 - cb, k_chunk / 16), P - p0);
        const int64_t nv = (np + 15) / 16;
        for (int64_t k = 0; k < Cin; ++k) {
            const float * src = xd + k * P + p0;
            for (int64_t v = 0; v < nv; ++v) {
                const __mmask16 m = 16 * v + 16 <= np ? (__mmask16) 0xffff : (__mmask16) ((1u << (np - 16 * v)) - 1);
                const __m512    a = _mm512_maskz_loadu_ps(m, src + 16 * v);
                _mm512_storeu_ps(xs.data() + (v * Cin + k) * 16,
                                 _mm512_cvtph_ps(_mm512_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)));
            }
        }
        for (int64_t v = 0; v < nv; ++v) {
            const float * xv = xs.data() + v * Cin * 16;
            // four output channels at a time share the input loads; their 16 accumulators each are computed four
            // at a time (accumulator m only ever takes channels m, m + 16, ... in order, so any grouping of the
            // m keeps the sums)
            for (int64_t oc0 = 0; oc0 < Cout; oc0 += 4) {
                const int64_t n_oc = std::min<int64_t>(4, Cout - oc0);
                __m512        a[4][16];
                for (int mg = 0; mg < 16; mg += 4) {
                    __m512 acc[4][4];
                    for (int o = 0; o < 4; ++o) {
                        for (int m = 0; m < 4; ++m) {
                            acc[o][m] = _mm512_setzero_ps();
                        }
                    }
                    for (int64_t l = 0; l < Cin; l += 16) {
                        for (int m = 0; m < 4; ++m) {
                            const __m512 xk = _mm512_loadu_ps(xv + (l + mg + m) * 16);
                            for (int o = 0; o < 4; ++o) {
                                const float * wr = wk.data() + std::min<int64_t>(oc0 + o, Cout - 1) * Cin;
                                acc[o][m]        = _mm512_fmadd_ps(xk, _mm512_set1_ps(wr[l + mg + m]), acc[o][m]);
                            }
                        }
                    }
                    for (int o = 0; o < 4; ++o) {
                        for (int m = 0; m < 4; ++m) {
                            a[o][mg + m] = acc[o][m];
                        }
                    }
                }
                for (int64_t o = 0; o < n_oc; ++o) {
                    const int64_t oc = oc0 + o;
                    __m512        t3[8];
                    for (int i = 0; i < 8; ++i) {
                        t3[i] = _mm512_add_ps(a[o][8 + i], a[o][i]);
                    }
                    __m512 t6[4];
                    for (int i = 0; i < 4; ++i) {
                        t6[i] = _mm512_add_ps(t3[4 + i], t3[i]);
                    }
                    __m512 r = _mm512_add_ps(_mm512_add_ps(t6[0], t6[2]), _mm512_add_ps(t6[1], t6[3]));
                    r        = _mm512_max_ps(_mm512_add_ps(r, _mm512_set1_ps(bd[oc])), _mm512_setzero_ps());
                    float * out = yd + oc * P + p0 + 16 * v;
                    if (16 * v + 16 <= np) {
                        _mm512_storeu_ps(out, r);
                    } else {
                        _mm512_mask_storeu_ps(out, (__mmask16) ((1u << (np - 16 * v)) - 1), r);
                    }
                }
            }
        }
    }
}

#endif  // GGML_CONV_FUSED_TINYBLAS

#if defined(GGML_CONV_FUSED_AVX512)

// The generic path: im2col (f32) and per channel a [9 x 1] matmul, each output a ggml_vec_dot_f32 of length 9,
// which GCC's AVX-512 build computes as one vector of the first eight products, added one by one to zero, and
// the ninth product added with an FMA; then a separate bias add. dot9 is that sum for one output.
inline float dot9(const float * w, const float * x) {
    __m128 s = _mm_setzero_ps();
    for (int k = 0; k < 8; ++k) {
        s = _mm_add_ss(s, _mm_mul_round_ss(_mm_set_ss(w[k]), _mm_set_ss(x[k]), _MM_FROUND_CUR_DIRECTION));
    }
    return _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(w[8]), _mm_set_ss(x[8]), s));
}

// Whether ggml_vec_dot_f32 of length 9 sums as dot9, on random values spanning many magnitudes
bool vec_dot9_matches() {
    static const bool ok = [] {
        std::mt19937                          rng(5);
        std::normal_distribution<float>       nd(0.0f, 1.0f);
        std::uniform_real_distribution<float> ue(-4.0f, 4.0f);
        float                                 w[9];
        float                                 x[9];
        for (int t = 0; t < 4096; ++t) {
            for (int k = 0; k < 9; ++k) {
                w[k] = nd(rng) * powf(10.0f, ue(rng));
                x[k] = nd(rng) * powf(10.0f, ue(rng));
            }
            float ref = 0.0f;
            ggml_vec_dot_f32(9, &ref, 0, w, 0, x, 0, 1);
            const float got = dot9(w, x);
            if (memcmp(&ref, &got, sizeof(float)) != 0) {
                return false;
            }
        }
        return true;
    }();
    return ok;
}

// The same sums with 16 neighbouring outputs of a row in the lanes, plus the bias.
void conv_2d_dw_s2_bias(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const ggml_tensor * x   = dst->src[0];
    const ggml_tensor * w   = dst->src[1];
    const ggml_tensor * b   = dst->src[2];
    const int           pad = (int) (intptr_t) userdata;

    const int64_t W  = x->ne[0];
    const int64_t H  = x->ne[1];
    const int64_t OW = dst->ne[0];
    const int64_t OH = dst->ne[1];
    const int64_t C  = dst->ne[2];

    const float * xd = (const float *) x->data;
    const float * wd = (const float *) w->data;
    const float * bd = (const float *) b->data;
    float *       yd = (float *) dst->data;

    const int64_t c0 = C * ith / nth;
    const int64_t c1 = C * (ith + 1) / nth;

    constexpr int64_t k_row = (k_max_width + 2 * 2) / 2 + 32;
    float             even[3][k_row];
    float             odd[3][k_row];
    float             buf[2 * k_row + 64] = {};
    const int64_t     n_vec = (OW + 15) / 16;

    for (int64_t c = c0; c < c1; ++c) {
        __m512 wv[9];
        for (int k = 0; k < 9; ++k) {
            wv[k] = _mm512_set1_ps(wd[c * 9 + k]);
        }
        const __m512  bc    = _mm512_set1_ps(bd[c]);
        const float * plane = xd + c * H * W;
        for (int64_t oh = 0; oh < OH; ++oh) {
            for (int kh = 0; kh < 3; ++kh) {
                const int64_t ih = 2 * oh - pad + kh;
                const bool    in = ih >= 0 && ih < H;
                split_row_even_odd(in ? plane + ih * W : plane, W, pad, in, n_vec, buf, even[kh], odd[kh]);
            }
            float * out = yd + (c * OH + oh) * OW;
            for (int64_t v = 0; v < n_vec; ++v) {
                const int64_t ow0 = 16 * v;
                __m512        s   = _mm512_setzero_ps();
                for (int kh = 0; kh < 3; ++kh) {
                    s = _mm512_add_ps(s, mul_unfused(wv[3 * kh + 0], _mm512_loadu_ps(even[kh] + ow0)));
                    s = _mm512_add_ps(s, mul_unfused(wv[3 * kh + 1], _mm512_loadu_ps(odd[kh] + ow0)));
                    if (kh < 2) {
                        s = _mm512_add_ps(s, mul_unfused(wv[3 * kh + 2], _mm512_loadu_ps(even[kh] + ow0 + 1)));
                    }
                }
                s = _mm512_fmadd_ps(wv[8], _mm512_loadu_ps(even[2] + ow0 + 1), s);
                s = _mm512_add_ps(s, bc);
                if (ow0 + 16 <= OW) {
                    _mm512_storeu_ps(out + ow0, s);
                } else {
                    _mm512_mask_storeu_ps(out + ow0, (__mmask16) ((1u << (OW - ow0)) - 1), s);
                }
            }
        }
    }
}

#if defined(__AVX512DQ__)

// The conv module of a conformer block after its gate, in the generic graph: transpose [C, T] -> [T, C],
// ggml_conv_2d_dw_direct along T (K = 9), bias add, BN as a multiply and an add, SiLU, transpose back. Its
// CONV_2D_DW op sums each output's valid taps in tap order; GCC's AVX-512 build multiplies taps 0-7 in one vector
// (invalid taps as zero products) and adds them one by one to zero, then adds tap 8 with an FMA. SiLU runs
// ggml_vec_silu_f32 over each channel's T values: ggml_v_silu for the first T & ~15, x / (1 + expf(-x)) for the
// rest. Here 16 channels share a vector, for one time step at a time, with the same operations per value.
constexpr int k_dw_taps = 9;

// tap sum of one output, as CONV_2D_DW computes it (valid taps j, at times t + j - pad)
inline float dw9_ref(const float * knl, const float * src, int64_t T, int64_t t, int pad) {
    __m128 s = _mm_setzero_ps();
    for (int j = 0; j < 8; ++j) {
        const int64_t st = t + j - pad;
        if (st >= 0 && st < T) {
            s = _mm_add_ss(s, _mm_mul_round_ss(_mm_set_ss(knl[j]), _mm_set_ss(src[st]), _MM_FROUND_CUR_DIRECTION));
        }
    }
    const int64_t st = t + 8 - pad;
    if (st >= 0 && st < T) {
        s = _mm_fmadd_ss(_mm_set_ss(knl[8]), _mm_set_ss(src[st]), s);
    }
    return _mm_cvtss_f32(s);
}

// Whether the CPU backend's CONV_2D_DW sums as dw9_ref, on random values spanning many magnitudes, with every
// output near both ends
bool conv_2d_dw_matches(int pad) {
    constexpr int64_t T = 40;
    constexpr int64_t C = 4;
    std::mt19937                          rng(9);
    std::normal_distribution<float>       nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ue(-4.0f, 4.0f);
    std::vector<float>                    src(T * C), knl(k_dw_taps * C), out(T * C);
    for (float & v : src) {
        v = nd(rng) * powf(10.0f, ue(rng));
    }
    for (float & v : knl) {
        v = nd(rng) * powf(10.0f, ue(rng));
    }
    ggml_tensor ts{};
    ggml_tensor tk{};
    ggml_tensor td{};
    ts.type  = GGML_TYPE_F32;
    ts.ne[0] = T;
    ts.ne[1] = 1;
    ts.ne[2] = C;
    ts.ne[3] = 1;
    tk.type  = GGML_TYPE_F32;
    tk.ne[0] = k_dw_taps;
    tk.ne[1] = 1;
    tk.ne[2] = 1;
    tk.ne[3] = C;
    td.type  = GGML_TYPE_F32;
    td.ne[0] = T + 2 * pad - k_dw_taps + 1;
    td.ne[1] = 1;
    td.ne[2] = C;
    td.ne[3] = 1;
    for (ggml_tensor * t : { &ts, &tk, &td }) {
        t->nb[0] = sizeof(float);
        for (int i = 1; i < 4; ++i) {
            t->nb[i] = t->nb[i - 1] * t->ne[i - 1];
        }
    }
    ts.data      = src.data();
    tk.data      = knl.data();
    td.data      = out.data();
    td.op        = GGML_OP_CONV_2D_DW;
    td.src[0]    = &tk;
    td.src[1]    = &ts;
    const int32_t op_params[6] = { 1, 1, pad, 0, 1, 1 };
    memcpy(td.op_params, op_params, sizeof(op_params));
    ggml_compute_params params{};
    params.ith = 0;
    params.nth = 1;
    ggml_compute_forward_conv_2d_dw(&params, &td);
    for (int64_t c = 0; c < C; ++c) {
        for (int64_t t = 0; t < td.ne[0]; ++t) {
            const float ref = out[c * td.ne[0] + t];
            const float got = dw9_ref(knl.data() + c * k_dw_taps, src.data() + c * T, T, t, pad);
            if (memcmp(&ref, &got, sizeof(float)) != 0) {
                return false;
            }
        }
    }
    return true;
}

void conv_1d_dw_bn_silu(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const ggml_tensor * x     = dst->src[0];
    const ggml_tensor * w     = dst->src[1];
    const ggml_tensor * scale = dst->src[2];
    const ggml_tensor * shift = dst->src[3];
    const ggml_tensor * b     = dst->src[4];
    const int           pad   = (int) (intptr_t) userdata;

    const int64_t C   = x->ne[0];
    const int64_t T   = x->ne[1];
    const int64_t T16 = T & ~(int64_t) 15;  // SiLU's vector part along each channel's row

    const float * xd = (const float *) x->data;
    const float * wd = (const float *) w->data;
    const float * sd = (const float *) scale->data;
    const float * hd = (const float *) shift->data;
    const float * bd = b != nullptr ? (const float *) b->data : nullptr;
    float *       yd = (float *) dst->data;

    // the kernel by tap, [9][C]
    static thread_local std::vector<float> kt;
    kt.resize(k_dw_taps * C);
    for (int64_t c = 0; c < C; ++c) {
        for (int j = 0; j < k_dw_taps; ++j) {
            kt[j * C + c] = wd[c * k_dw_taps + j];
        }
    }

    const int64_t t0  = T * ith / nth;
    const int64_t t1  = T * (ith + 1) / nth;
    const __m512  one = _mm512_set1_ps(1.0f);
    for (int64_t t = t0; t < t1; ++t) {
        float * out = yd + t * C;
        for (int64_t c = 0; c < C; c += 16) {
            __m512 s = _mm512_setzero_ps();
            for (int j = 0; j < 8; ++j) {
                const int64_t st = t + j - pad;
                if (st >= 0 && st < T) {
                    s = _mm512_add_ps(s, mul_unfused(_mm512_loadu_ps(kt.data() + j * C + c), _mm512_loadu_ps(xd + st * C + c)));
                }
            }
            const int64_t st = t + 8 - pad;
            if (st >= 0 && st < T) {
                s = _mm512_fmadd_ps(_mm512_loadu_ps(kt.data() + 8 * C + c), _mm512_loadu_ps(xd + st * C + c), s);
            }
            if (bd != nullptr) {
                s = _mm512_add_ps(s, _mm512_loadu_ps(bd + c));
            }
            s = _mm512_add_ps(mul_unfused(s, _mm512_loadu_ps(sd + c)), _mm512_loadu_ps(hd + c));
            if (t < T16) {
                s = ggml_v_silu(s);
            } else {
                s = _mm512_div_ps(s, _mm512_add_ps(one, ggml_v_expf_f64(_mm512_sub_ps(_mm512_setzero_ps(), s))));
            }
            _mm512_storeu_ps(out + c, s);
        }
    }
}

#endif  // __AVX512DQ__

#if defined(__AVX512BW__) && defined(__AVX512VL__)

// ggml_vec_dot_f32 of length D, a multiple of 64, in GCC's AVX-512 build: four vector accumulators taking the
// products of 16-value blocks j, j + 4, j + 8, ... by FMA (from zero), combined as (a0 + a2) + (a1 + a3), then
// _mm512_reduce_add_ps. dot_lanes returns the combined vector before that last reduction.
inline __m512 dot_lanes(const float * x, const __m512 * y, int64_t D) {
    __m512 a[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
    for (int64_t i = 0; i < D; i += 64) {
        for (int j = 0; j < 4; ++j) {
            a[j] = _mm512_fmadd_ps(_mm512_loadu_ps(x + i + 16 * j), y[i / 16 + j], a[j]);
        }
    }
    return _mm512_add_ps(_mm512_add_ps(a[0], a[2]), _mm512_add_ps(a[1], a[3]));
}

// _mm512_reduce_add_ps of 16 vectors at once, lane o of the result being the reduction of u[o], with GCC's tree
// for each: t3[i] = u[8 + i] + u[i], t6[i] = t3[4 + i] + t3[i], (t6[0] + t6[2]) + (t6[1] + t6[3]).
// reduce_pair is the first step (t3 of two vectors), reduce_add_w8 the rest from the eight pairs.
inline __m512 reduce_pair(__m512 u0, __m512 u1) {  // 8 + 8 lanes: t3 of u0, u1
    return _mm512_add_ps(_mm512_shuffle_f32x4(u0, u1, 0xee), _mm512_shuffle_f32x4(u0, u1, 0x44));
}

inline __m512 reduce_add_w8(const __m512 * w) {
    __m512 z[4];
    for (int k = 0; k < 4; ++k) {  // 4 x 4 lanes: t6 of u[4k] .. u[4k + 3]
        z[k] = _mm512_add_ps(_mm512_shuffle_f32x4(w[2 * k], w[2 * k + 1], 0xdd),
                             _mm512_shuffle_f32x4(w[2 * k], w[2 * k + 1], 0x88));
    }
    __m512 v[2];
    for (int k = 0; k < 2; ++k) {  // per 128-bit lane L: t6[0] + t6[2], t6[1] + t6[3] of z[2k] and z[2k + 1] lane L
        v[k] = _mm512_add_ps(_mm512_shuffle_ps(z[2 * k], z[2 * k + 1], 0x44), _mm512_shuffle_ps(z[2 * k], z[2 * k + 1], 0xee));
    }
    // lane 4L + s: the result of u[4s + L]; reorder to lane o = u[o]
    const __m512  r   = _mm512_add_ps(_mm512_shuffle_ps(v[0], v[1], 0x88), _mm512_shuffle_ps(v[0], v[1], 0xdd));
    const __m512i idx = _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    return _mm512_permutexvar_ps(idx, r);
}

inline __m512 reduce_add_x16(const __m512 * u) {
    __m512 w[8];
    for (int k = 0; k < 8; ++k) {
        w[k] = reduce_pair(u[2 * k], u[2 * k + 1]);
    }
    return reduce_add_w8(w);
}

// Whether ggml_vec_dot_f32 of length D sums as dot_lanes + _mm512_reduce_add_ps, and reduce_add_x16 reduces as
// _mm512_reduce_add_ps, on random values spanning many magnitudes
bool vec_dot_rows_match(int64_t D) {
    std::mt19937                          rng(3);
    std::normal_distribution<float>       nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ue(-4.0f, 4.0f);
    std::vector<float>                    x(16 * D), y(D);
    for (int t = 0; t < 64; ++t) {
        for (float & v : x) {
            v = nd(rng) * powf(10.0f, ue(rng));
        }
        for (float & v : y) {
            v = nd(rng) * powf(10.0f, ue(rng));
        }
        std::vector<__m512> yv(D / 16);
        for (int64_t i = 0; i < D / 16; ++i) {
            yv[i] = _mm512_loadu_ps(y.data() + 16 * i);
        }
        __m512 u[16];
        for (int o = 0; o < 16; ++o) {
            u[o] = dot_lanes(x.data() + o * D, yv.data(), D);
        }
        float got[16];
        _mm512_storeu_ps(got, reduce_add_x16(u));
        for (int o = 0; o < 16; ++o) {
            float       ref = 0.0f;
            const float one = _mm512_reduce_add_ps(u[o]);
            ggml_vec_dot_f32((int) D, &ref, 0, x.data() + o * D, 0, y.data(), 0, 1);
            if (memcmp(&ref, &got[o], sizeof(float)) != 0 || memcmp(&ref, &one, sizeof(float)) != 0) {
                return false;
            }
        }
    }
    return true;
}

// dot_lanes for a D known at compile time, fully unrolled
template <int64_t DC> __attribute__((always_inline)) inline __m512 dot_lanes_c(const float * x, const __m512 * y) {
    __m512 a[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
#pragma GCC unroll 16
    for (int64_t i = 0; i < DC; i += 64) {
#pragma GCC unroll 4
        for (int j = 0; j < 4; ++j) {
            a[j] = _mm512_fmadd_ps(_mm512_loadu_ps(x + i + 16 * j), y[i / 16 + j], a[j]);
        }
    }
    return _mm512_add_ps(_mm512_add_ps(a[0], a[2]), _mm512_add_ps(a[1], a[3]));
}

// Scores of the queries i0 .. i0 + nq - 1 of one head against positions packed contiguously: rows[k] is position
// lo + k, with 16 more rows after the last position the block takes. Query i takes positions T - 1 - i + j for
// j = 0 .. T - 1. The positions are visited in tiles of 16 that all queries of the block use in turn while the tile
// is in L1; each score is dot_lanes + reduce_add_x16 as before (lanes outside 0 .. T - 1 are computed from other
// rows and not stored). DC is D when known at compile time (0: use D).
template <int64_t DC>
void rel_pos_query_block(const float * rows, int64_t lo, const ggml_tensor * q, int64_t h, int64_t i0, int64_t nq,
                         int64_t T, int64_t D_rt, __m512 sv, ggml_tensor * dst) {
    const int64_t D     = DC > 0 ? DC : D_rt;
    // positions of the block: T - i0 - nq .. 2T - 2 - i0
    const int64_t p_beg = T - i0 - nq;
    const int64_t p_end = 2 * T - 1 - i0;
    __m512        yv[DC > 0 ? DC / 16 : 64];
    for (int64_t pt = p_beg; pt < p_end; pt += 16) {
        for (int64_t i = i0; i < i0 + nq; ++i) {
            // j of position pt for query i; the tile covers j0 .. j0 + 15, of which 0 .. T - 1 exist
            const int64_t j0 = pt - (T - 1 - i);
            const int64_t a  = std::max<int64_t>(0, -j0);
            const int64_t b  = std::min<int64_t>(16, T - j0);
            if (a >= b) {
                continue;
            }
            const float * qr = (const float *) ((const char *) q->data + i * q->nb[1] + h * q->nb[2]);
            for (int64_t k = 0; k < D / 16; ++k) {
                yv[k] = _mm512_loadu_ps(qr + 16 * k);
            }
            const float * pr = rows + (pt - lo) * D;
            __m512        w[8];
            if (DC > 0) {
#pragma GCC unroll 8
                for (int k = 0; k < 8; ++k) {
                    const __m512 u0 = dot_lanes_c<DC>(pr + 2 * k * DC, yv);
                    const __m512 u1 = dot_lanes_c<DC>(pr + (2 * k + 1) * DC, yv);
                    w[k]            = reduce_pair(u0, u1);
                }
            } else {
                for (int k = 0; k < 8; ++k) {
                    w[k] = reduce_pair(dot_lanes(pr + 2 * k * D, yv, D), dot_lanes(pr + (2 * k + 1) * D, yv, D));
                }
            }
            const __m512    sc  = _mm512_mul_ps(reduce_add_w8(w), sv);
            const __m256i   h16 = _mm512_cvtps_ph(sc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            ggml_fp16_t *   out = (ggml_fp16_t *) ((char *) dst->data + i * dst->nb[1] + h * dst->nb[2]);
            const __mmask16 m   = (__mmask16) (((1u << b) - 1) & ~((1u << a) - 1));
            _mm256_mask_storeu_epi16(out + j0, m, h16);
        }
    }
}

void rel_pos_scores_f16(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * p = dst->src[1];
    float               scale;
    const uint32_t      bits = (uint32_t) (uintptr_t) userdata;
    memcpy(&scale, &bits, sizeof(float));

    const int64_t D = q->ne[0];
    const int64_t T = q->ne[1];
    const int64_t H = q->ne[2];

    // queries per pass over a tile of positions
    constexpr int64_t k_query_block = 8;

    const __m512                           sv    = _mm512_set1_ps(scale);
    const int64_t                          n_row = H * T;
    const int64_t                          r0    = n_row * ith / nth;
    const int64_t                          r1    = n_row * (ith + 1) / nth;
    static thread_local std::vector<float> rows;
    for (int64_t r = r0; r < r1;) {
        // this thread's queries i0 .. i1 - 1 of head h, and the positions they take, T - i1 .. 2T - 2 - i0, copied
        // to consecutive rows (in the projection's layout consecutive positions of a head are H rows apart)
        const int64_t h  = r / T;
        const int64_t i0 = r % T;
        const int64_t i1 = std::min(T, i0 + (r1 - r));
        const int64_t lo = T - i1;
        const int64_t n  = T + (i1 - i0) - 1;
        if ((int64_t) rows.size() < (n + 16) * D) {
            rows.assign((n + 16) * D, 0.0f);
        }
        for (int64_t k = 0; k < n; ++k) {
            memcpy(rows.data() + k * D, (const char *) p->data + h * p->nb[2] + (lo + k) * p->nb[1], D * sizeof(float));
        }
        for (int64_t i = i0; i < i1; i += k_query_block) {
            const int64_t nq = std::min(k_query_block, i1 - i);
            if (D == 128) {
                rel_pos_query_block<128>(rows.data(), lo, q, h, i, nq, T, D, sv, dst);
            } else if (D == 64) {
                rel_pos_query_block<64>(rows.data(), lo, q, h, i, nq, T, D, sv, dst);
            } else {
                rel_pos_query_block<0>(rows.data(), lo, q, h, i, nq, T, D, sv, dst);
            }
        }
        r += i1 - i0;
    }
}

#endif  // __AVX512BW__ && __AVX512VL__

#endif  // GGML_CONV_FUSED_AVX512

}  // namespace

extern "C" ggml_custom_op_t ggml_cpu_conv_2d_s2_c1_bias_relu_op(const ggml_tensor * x, const ggml_tensor * w,
                                                                const ggml_tensor * b, int pad) {
#if defined(GGML_CONV_FUSED_TINYBLAS)
    if (x == nullptr || w == nullptr || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[2] != 1 ||
        x->ne[3] != 1 || x->ne[0] > k_max_width) {
        return nullptr;
    }
    if ((w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_F16) || !ggml_is_contiguous(w) || w->ne[0] != 3 ||
        w->ne[1] != 3 || w->ne[2] != 1 || w->ne[3] < 2 || w->ne[3] > k_max_channels || !is_f32_vector(b, w->ne[3])) {
        return nullptr;
    }
    const int64_t OW = conv_out_size_k3s2(x->ne[0], pad);
    const int64_t OH = conv_out_size_k3s2(x->ne[1], pad);
    // the generic matmul takes the tinyBLAS kernel only when its rows divide by 4
    if (pad < 0 || pad > 2 || OW < 1 || OH < 1 || (OW * OH) % 4 != 0) {
        return nullptr;
    }
    return conv_2d_s2_c1_bias_relu;
#else
    (void) x;
    (void) w;
    (void) b;
    (void) pad;
    return nullptr;
#endif
}

extern "C" ggml_custom_op_t ggml_cpu_conv_2d_pw_bias_relu_op(const ggml_tensor * x, const ggml_tensor * w,
                                                             const ggml_tensor * b, int pad) {
#if defined(GGML_CONV_FUSED_TINYBLAS)
    if (x == nullptr || w == nullptr || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[3] != 1 ||
        pad != 0) {
        return nullptr;
    }
    const int64_t Cin = x->ne[2];
    if ((w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_F16) || !ggml_is_contiguous(w) || w->ne[0] != 1 ||
        w->ne[1] != 1 || w->ne[2] != Cin || w->ne[3] < 2 || !is_f32_vector(b, w->ne[3])) {
        return nullptr;
    }
    // the generic matmul takes the tinyBLAS kernel, without a partial vector, only for rows that divide by 4 and
    // a channel count that divides by 16
    if (Cin % 16 != 0 || Cin > k_max_channels || (x->ne[0] * x->ne[1]) % 4 != 0) {
        return nullptr;
    }
    return conv_2d_pw_bias_relu;
#else
    (void) x;
    (void) w;
    (void) b;
    (void) pad;
    return nullptr;
#endif
}

extern "C" ggml_custom_op_t ggml_cpu_conv_1d_dw_bn_silu_op(const ggml_tensor * x, const ggml_tensor * w,
                                                           const ggml_tensor * b, const ggml_tensor * scale,
                                                           const ggml_tensor * shift, int pad) {
#if defined(GGML_CONV_FUSED_AVX512) && defined(__AVX512DQ__)
    if (x == nullptr || w == nullptr || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[2] != 1 ||
        x->ne[3] != 1 || x->ne[0] % 16 != 0) {
        return nullptr;
    }
    const int64_t C = x->ne[0];
    if (w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || ggml_nelements(w) != k_dw_taps * C ||
        w->ne[0] != k_dw_taps || !is_f32_vector(scale, C) || !is_f32_vector(shift, C) ||
        (b != nullptr && !is_f32_vector(b, C))) {
        return nullptr;
    }
    // output length equal to the input's: symmetric padding (K - 1) / 2
    if (pad != (k_dw_taps - 1) / 2) {
        return nullptr;
    }
    static const bool ok = conv_2d_dw_matches((k_dw_taps - 1) / 2);
    return ok ? conv_1d_dw_bn_silu : nullptr;
#else
    (void) x;
    (void) w;
    (void) b;
    (void) scale;
    (void) shift;
    (void) pad;
    return nullptr;
#endif
}

extern "C" ggml_custom_op_t ggml_cpu_rel_pos_scores_f16_op(const ggml_tensor * q, const ggml_tensor * p) {
#if defined(GGML_CONV_FUSED_AVX512) && defined(__AVX512BW__) && defined(__AVX512VL__)
    if (q == nullptr || p == nullptr || q->type != GGML_TYPE_F32 || p->type != GGML_TYPE_F32 ||
        q->nb[0] != sizeof(float) || p->nb[0] != sizeof(float) || q->ne[3] != 1 || p->ne[3] != 1) {
        return nullptr;
    }
    const int64_t D = q->ne[0];
    const int64_t T = q->ne[1];
    // the generic matmul runs ggml_vec_dot_f32 (not tinyBLAS) as its 2T - 1 rows do not divide by 4, without
    // leftover products for D a multiple of 64
    if (D % 64 != 0 || D > 1024 || p->ne[0] != D || p->ne[1] != 2 * T - 1 || p->ne[2] != q->ne[2]) {
        return nullptr;
    }
    static const bool ok = vec_dot_rows_match(128) && vec_dot_rows_match(64);
    return ok ? rel_pos_scores_f16 : nullptr;
#else
    (void) q;
    (void) p;
    return nullptr;
#endif
}

extern "C" ggml_custom_op_t ggml_cpu_conv_2d_dw_s2_bias_op(const ggml_tensor * x, const ggml_tensor * w,
                                                           const ggml_tensor * b, int pad) {
#if defined(GGML_CONV_FUSED_AVX512)
    if (x == nullptr || w == nullptr || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[3] != 1 ||
        x->ne[0] > k_max_width) {
        return nullptr;
    }
    if (w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || w->ne[0] != 3 || w->ne[1] != 3 || w->ne[2] != 1 ||
        w->ne[3] != x->ne[2] || !is_f32_vector(b, w->ne[3])) {
        return nullptr;
    }
    if (pad < 0 || pad > 2 || conv_out_size_k3s2(x->ne[0], pad) < 1 || conv_out_size_k3s2(x->ne[1], pad) < 1 ||
        !vec_dot9_matches()) {
        return nullptr;
    }
    return conv_2d_dw_s2_bias;
#else
    (void) x;
    (void) w;
    (void) b;
    (void) pad;
    return nullptr;
#endif
}
