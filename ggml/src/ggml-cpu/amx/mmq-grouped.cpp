#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

#include "mmq-grouped.h"
#include "amx.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cstring>

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)

#include <immintrin.h>

#if (defined(_WIN32) || defined(_WIN64))
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

namespace {

constexpr uint64_t AMX_Q8_0_GROUPED_MAGIC = 0x31505247305138ull;  // "8Q0GRP1"
constexpr int      G_TILE                 = TILE_M * 64;          // bytes of one 16 x 64 int8 tile
constexpr int      G_RING                 = 3;                    // stored running totals: two groups + the next

// activation group for weights converted on this thread: -1 off, 0 one scale per row, > 0 values per scale
thread_local int g_load_act_group = -1;

// all eight tiles 16 rows x 64 bytes: A and B tiles hold 64 int8 values per row, C tiles 16 int32
void grouped_tile_config() {
    if (ggml_amx_tile_config_id == AMX_TILE_CONFIG_GROUPED) {
        return;
    }
    alignas(64) tile_config_t tc = {};
    tc.palette_id                = 1;
    for (int t = 0; t < 8; ++t) {
        tc.rows[t]  = TILE_M;
        tc.colsb[t] = 64;
    }
    _tile_loadconfig(&tc);
    ggml_amx_tile_config_id = AMX_TILE_CONFIG_GROUPED;
}

// the largest of (K, 512, 256, 128, 64) values per group in which every row's blocks share one scale, or 0
int detect_weight_group(const block_q8_0 * B, int N, int K) {
    const int KB      = K / QK8_0;
    const int cand[5] = { K, 512, 256, 128, 64 };
    for (int G : cand) {
        if (G > K || K % G != 0 || G % 64 != 0) {
            continue;
        }
        const int nb = G / QK8_0;
        bool      ok = true;
        for (int64_t n = 0; n < N && ok; ++n) {
            const block_q8_0 * row = B + n * KB;
            for (int kb = 0; kb < KB; ++kb) {
                if (row[kb].d != row[kb - kb % nb].d) {
                    ok = false;
                    break;
                }
            }
        }
        if (ok) {
            return G;
        }
    }
    return 0;
}

// one activation row to int8 with one scale per `ag` values (amax / 127, as Q8_0 computes it). Tile order: value k
// goes to a[(k / 64) * G_TILE + k % 64], the row's place in its 16-row tile block. Plain order (the VNNI kernels):
// a[k] = the value + 128, as uint8.
template <bool PLAIN>
void quantize_row(const float * RESTRICT x, int8_t * RESTRICT a, float * RESTRICT as, int K, int ag) {
    const __m512 sign = _mm512_set1_ps(-0.0f);
    for (int g0 = 0; g0 < K; g0 += ag) {
        __m512 m0 = _mm512_setzero_ps();
        __m512 m1 = _mm512_setzero_ps();
        __m512 m2 = _mm512_setzero_ps();
        __m512 m3 = _mm512_setzero_ps();
        for (int k = g0; k < g0 + ag; k += 64) {
            m0 = _mm512_max_ps(m0, _mm512_andnot_ps(sign, _mm512_loadu_ps(x + k)));
            m1 = _mm512_max_ps(m1, _mm512_andnot_ps(sign, _mm512_loadu_ps(x + k + 16)));
            m2 = _mm512_max_ps(m2, _mm512_andnot_ps(sign, _mm512_loadu_ps(x + k + 32)));
            m3 = _mm512_max_ps(m3, _mm512_andnot_ps(sign, _mm512_loadu_ps(x + k + 48)));
        }
        const float d  = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_max_ps(m0, m1), _mm512_max_ps(m2, m3))) / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        as[g0 / ag]    = d;
        const __m512 vid = _mm512_set1_ps(id);
        for (int k = g0; k < g0 + ag; k += 64) {
            __m128i q[4];
            for (int j = 0; j < 4; ++j) {
                const __m512i v = _mm512_cvt_roundps_epi32(_mm512_mul_ps(_mm512_loadu_ps(x + k + 16 * j), vid),
                                                           _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                q[j]            = _mm512_cvtsepi32_epi8(v);
            }
            __m512i v = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set_m128i(q[1], q[0])),
                                           _mm256_set_m128i(q[3], q[2]), 1);
            if constexpr (PLAIN) {
                _mm512_storeu_si512((__m512i *) (a + k), _mm512_xor_si512(v, _mm512_set1_epi8((char) 0x80)));
            } else {
                _mm512_storeu_si512((__m512i *) (a + (size_t) (unsigned) (k >> 6) * G_TILE), v);
            }
        }
    }
}

// scaling groups: R values per scaled add (the smaller of the weight and activation groups), NR per row, SR tile
// steps of 64 values each
struct group_scales {
    int R, NR, SR, NGw, NGa, wg, ag;

    explicit group_scales(const amx_q8_0_grouped_info & gi) :
        R(std::min(gi.wg, gi.ag)),
        NR(gi.K / R),
        SR(R / 64),
        NGw(gi.K / gi.wg),
        NGa(gi.K / gi.ag),
        wg(gi.wg),
        ag(gi.ag) {}
};

// 16 outputs of one row: c (+)= group sum x weight scales w x activation scale a. The group sum is cur, the running
// total after this group, minus prev, the total after the group before (none for the first group).
inline void acc_row(float * RESTRICT c, const int32_t * RESTRICT cur, const int32_t * RESTRICT prev, __m512 w,
                    float a, bool first) {
    __m512i v = _mm512_load_si512(cur);
    if (!first) {
        v = _mm512_sub_epi32(v, _mm512_load_si512(prev));
    }
    const __m512 s = _mm512_mul_ps(w, _mm512_set1_ps(a));
    const __m512 f = _mm512_cvtepi32_ps(v);
    _mm512_store_ps(c, first ? _mm512_mul_ps(f, s) : _mm512_fmadd_ps(f, s, _mm512_load_ps(c)));
}

inline void write_row(float * RESTRICT dst, const float * RESTRICT c, const float * RESTRICT bias, int n16) {
    for (int t = 0; t < n16; ++t) {
        __m512 v = _mm512_load_ps(c + t * 16);
        if (bias != nullptr) {
            // the bias add of a fused MUL_MAT + ADD: one rounding, as the ADD node
            v = _mm512_add_ps(v, _mm512_loadu_ps(bias + t * 16));
        }
        _mm512_storeu_ps(dst + t * 16, v);
    }
}

// ROWS (1..AMX_GROUPED_VNNI_ROWS) rows x NT weight tiles from column tile nt0 with AVX512-VNNI, which streams the
// weights with ordinary loads (the tile loads of the AMX kernels keep too few of them in flight when the weights
// come from memory). A: rows of K activations + 128 as uint8 (row stride K); each group's sums are
// sum((a + 128) * w) - 128 * sum(w), the column sums made at load (wsum).
constexpr int AMX_GROUPED_VNNI_ROWS = 8;

template <int ROWS, int NT>
void grouped_vnni_rows(const amx_q8_0_grouped_info & gi, const group_scales & gs, const int8_t * RESTRICT W,
                       const float * RESTRICT Ws, const int32_t * RESTRICT Wsum, const uint8_t * RESTRICT A,
                       const float * RESTRICT as, int nt0, float * RESTRICT C, int ldc, const float * RESTRICT bias) {
    const int K  = gi.K;
    const int KS = K / 64;
    __m512    vc[ROWS][NT];
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < NT; ++c) {
            vc[r][c] = _mm512_setzero_ps();
        }
    }
    for (int g = 0; g < gs.NR; ++g) {
        const int gw = g * gs.R / gs.wg;
        const int ga = g * gs.R / gs.ag;
        for (int c = 0; c < NT; ++c) {
            const int nt = nt0 + c;
            __m512i   acc[2][ROWS];
            for (int r = 0; r < ROWS; ++r) {
                acc[0][r] = _mm512_setzero_si512();
                acc[1][r] = _mm512_setzero_si512();
            }
            for (int s = 0; s < gs.SR; ++s) {
                const int      ks = g * gs.SR + s;
                const int8_t * w  = W + ((size_t) nt * KS + ks) * G_TILE;
                const uint8_t * a = A + (size_t) ks * 64;
#pragma GCC unroll 16
                for (int j = 0; j < 16; ++j) {
                    const __m512i vw = _mm512_loadu_si512((const __m512i *) (w + 64 * j));
                    for (int r = 0; r < ROWS; ++r) {
                        int32_t av;
                        memcpy(&av, a + (size_t) r * K + 4 * j, sizeof(av));
                        acc[j % 2][r] = _mm512_dpbusd_epi32(acc[j % 2][r], _mm512_set1_epi32(av), vw);
                    }
                }
            }
            const __m512  ws   = _mm512_loadu_ps(Ws + ((size_t) nt * gs.NGw + gw) * TILE_N);
            const __m512i corr = _mm512_loadu_si512((const __m512i *) (Wsum + ((size_t) nt * gs.NR + g) * TILE_N));
            for (int r = 0; r < ROWS; ++r) {
                const __m512i sum = _mm512_sub_epi32(_mm512_add_epi32(acc[0][r], acc[1][r]), corr);
                vc[r][c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sum), _mm512_mul_ps(ws, _mm512_set1_ps(as[(size_t) r * gs.NGa + ga])),
                                           vc[r][c]);
            }
        }
    }
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < NT; ++c) {
            __m512 v = vc[r][c];
            if (bias != nullptr) {
                v = _mm512_add_ps(v, _mm512_loadu_ps(bias + c * TILE_N));
            }
            _mm512_storeu_ps(C + (size_t) r * ldc + c * TILE_N, v);
        }
    }
}

template <int ROWS>
void grouped_vnni_dispatch(const amx_q8_0_grouped_info & gi, const group_scales & gs, int rows, int nt,
                           const int8_t * W, const float * Ws, const int32_t * Wsum, const uint8_t * A,
                           const float * as, int nt0, float * C, int ldc, const float * bias) {
    if constexpr (ROWS > 1) {
        if (rows < ROWS) {
            grouped_vnni_dispatch<ROWS - 1>(gi, gs, rows, nt, W, Ws, Wsum, A, as, nt0, C, ldc, bias);
            return;
        }
    }
    if (nt == 4) {
        grouped_vnni_rows<ROWS, 4>(gi, gs, W, Ws, Wsum, A, as, nt0, C, ldc, bias);
    } else {
        grouped_vnni_rows<ROWS, 2>(gi, gs, W, Ws, Wsum, A, as, nt0, C, ldc, bias);
    }
}

// one 32-column panel (column tiles 2 nb, 2 nb + 1) for all rows, two 16-row A tiles at a time. Tiles: B 0-1,
// A 2-3, C 4 / 6 (first A tile x B 0 / 1) and 5 / 7 (second A tile).
void grouped_panel(const amx_q8_0_grouped_info & gi, const group_scales & gs, const int8_t * RESTRICT W,
                   const float * RESTRICT Ws, const int8_t * RESTRICT A, const float * RESTRICT as, int M, int nb,
                   float * RESTRICT C, int ldc, const float * RESTRICT bias) {
    const int      KS  = gi.K / 64;
    const size_t   TS  = (size_t) KS * G_TILE;
    const int      MT  = (M + TILE_M - 1) / TILE_M;
    const int8_t * B0  = W + (size_t) (2 * nb) * TS;
    const int8_t * B1  = B0 + TS;
    const float *  ws0 = Ws + (size_t) (2 * nb) * gs.NGw * TILE_N;
    const float *  ws1 = ws0 + (size_t) gs.NGw * TILE_N;
    alignas(64) int32_t ring[G_RING][4][TILE_M * TILE_N];
    alignas(64) float   cb[2 * TILE_M][2 * TILE_N];

    for (int mt = 0; mt < MT; mt += 2) {
        const bool     two = mt + 1 < MT;
        const int      nr0 = std::min(TILE_M, M - mt * TILE_M);
        const int      nr1 = two ? std::min(TILE_M, M - (mt + 1) * TILE_M) : 0;
        const int8_t * A0  = A + (size_t) mt * TS;
        const int8_t * A1  = A0 + TS;
        const float *  s0  = as + (size_t) mt * TILE_M * gs.NGa;
        const float *  s1  = s0 + (size_t) TILE_M * gs.NGa;

        auto consume = [&](int g) {
            const int       gw   = g * gs.R / gs.wg;
            const int       ga   = g * gs.R / gs.ag;
            const __m512    w0   = _mm512_loadu_ps(ws0 + (size_t) gw * TILE_N);
            const __m512    w1   = _mm512_loadu_ps(ws1 + (size_t) gw * TILE_N);
            const int32_t(*cur)[TILE_M * TILE_N]  = ring[g % G_RING];
            const int32_t(*prev)[TILE_M * TILE_N] = ring[(g + G_RING - 1) % G_RING];
            const bool first = g == 0;
            for (int r = 0; r < nr0; ++r) {
                const float a = s0[(size_t) r * gs.NGa + ga];
                acc_row(&cb[r][0], cur[0] + r * TILE_N, prev[0] + r * TILE_N, w0, a, first);
                acc_row(&cb[r][TILE_N], cur[2] + r * TILE_N, prev[2] + r * TILE_N, w1, a, first);
            }
            for (int r = 0; r < nr1; ++r) {
                const float a = s1[(size_t) r * gs.NGa + ga];
                acc_row(&cb[TILE_M + r][0], cur[1] + r * TILE_N, prev[1] + r * TILE_N, w0, a, first);
                acc_row(&cb[TILE_M + r][TILE_N], cur[3] + r * TILE_N, prev[3] + r * TILE_N, w1, a, first);
            }
        };

        _tile_zero(TMM4);
        _tile_zero(TMM6);
        if (two) {
            _tile_zero(TMM5);
            _tile_zero(TMM7);
        }
        for (int g = 0; g < gs.NR; ++g) {
            for (int s = 0; s < gs.SR; ++s) {
                const size_t ko = (size_t) (g * gs.SR + s) * G_TILE;
                _tile_loadd(TMM2, A0 + ko, 64);
                _tile_loadd(TMM0, B0 + ko, 64);
                _tile_dpbssd(TMM4, TMM2, TMM0);
                _tile_loadd(TMM1, B1 + ko, 64);
                _tile_dpbssd(TMM6, TMM2, TMM1);
                if (two) {
                    _tile_loadd(TMM3, A1 + ko, 64);
                    _tile_dpbssd(TMM5, TMM3, TMM0);
                    _tile_dpbssd(TMM7, TMM3, TMM1);
                }
            }
            int32_t(*cur)[TILE_M * TILE_N] = ring[g % G_RING];
            _tile_stored(TMM4, cur[0], 64);
            _tile_stored(TMM6, cur[2], 64);
            if (two) {
                _tile_stored(TMM5, cur[1], 64);
                _tile_stored(TMM7, cur[3], 64);
            }
            if (g > 0) {
                consume(g - 1);
            }
        }
        consume(gs.NR - 1);
        for (int r = 0; r < nr0 + nr1; ++r) {
            write_row(C + (size_t) (mt * TILE_M + r) * ldc, cb[r], bias, 2);
        }
    }
}

}  // namespace

void ggml_backend_amx_set_q8_0_grouped(int act_group) {
    g_load_act_group = act_group;
}

void amx_q8_0_grouped_clear(void * trailer) {
    memset(trailer, 0, AMX_Q8_0_TRAILER_SIZE);
}

bool amx_q8_0_grouped_pack(const struct ggml_tensor * t, const void * data, void * trailer) {
    const int ag_req = g_load_act_group;
    if (ag_req < 0 || t->type != GGML_TYPE_Q8_0) {
        return false;
    }
    const int K = t->ne[0];
    const int N = t->ne[1];
    if (ggml_nrows(t) != N || K % 64 != 0 || N % (2 * TILE_N) != 0) {
        return false;
    }
    const block_q8_0 * B  = (const block_q8_0 *) data;
    const int          wg = detect_weight_group(B, N, K);
    if (wg == 0) {
        return false;
    }
    int ag = ag_req == 0 ? K : std::min(ag_req, K);
    if (ag % 64 != 0 || K % ag != 0 || std::max(ag, wg) % std::min(ag, wg) != 0) {
        ag = wg;
    }
    const int KS  = K / 64;
    const int KB  = K / QK8_0;
    const int NGw = K / wg;
    int8_t *  q   = (int8_t *) t->data;
    float *   s   = (float *) (q + (size_t) N * K);
    parallel_for(N / TILE_N, [&](int begin, int end) {
        for (int nt = begin; nt < end; ++nt) {
            for (int c = 0; c < TILE_N; ++c) {
                const block_q8_0 * row = B + (size_t) (nt * TILE_N + c) * KB;
                for (int g = 0; g < NGw; ++g) {
                    s[((size_t) nt * NGw + g) * TILE_N + c] = GGML_CPU_FP16_TO_FP32(row[g * (wg / QK8_0)].d);
                }
                int8_t * tile = q + (size_t) nt * KS * G_TILE;
                for (int k = 0; k < K; ++k) {
                    tile[(size_t) (k / 64) * G_TILE + ((k % 64) / 4) * 64 + c * 4 + k % 4] = row[k / QK8_0].qs[k % QK8_0];
                }
            }
        }
    });
    // 128 x the column sums of each scaling group, for the VNNI kernels' unsigned activations
    const int R    = std::min(wg, ag);
    const int NR   = K / R;
    int32_t * wsum = (int32_t *) (s + (size_t) N * NGw);
    parallel_for(N / TILE_N, [&](int begin, int end) {
        for (int nt = begin; nt < end; ++nt) {
            for (int c = 0; c < TILE_N; ++c) {
                const block_q8_0 * row = B + (size_t) (nt * TILE_N + c) * KB;
                for (int g = 0; g < NR; ++g) {
                    int32_t sum = 0;
                    for (int k = g * R; k < (g + 1) * R; ++k) {
                        sum += row[k / QK8_0].qs[k % QK8_0];
                    }
                    wsum[((size_t) nt * NR + g) * TILE_N + c] = 128 * sum;
                }
            }
        }
    });
    amx_q8_0_grouped_info gi = { AMX_Q8_0_GROUPED_MAGIC, K, N, wg, ag };
    amx_q8_0_grouped_clear(trailer);
    memcpy(trailer, &gi, sizeof(gi));
    return true;
}

const amx_q8_0_grouped_info * amx_q8_0_grouped_get(const struct ggml_tensor * src0, const void * trailer) {
    const auto * gi = (const amx_q8_0_grouped_info *) trailer;
    if (gi->magic != AMX_Q8_0_GROUPED_MAGIC || gi->K != src0->ne[0] || gi->N != src0->ne[1]) {
        return nullptr;
    }
    return gi;
}

size_t amx_q8_0_grouped_wsize(const amx_q8_0_grouped_info & gi, int64_t M) {
    const int64_t rows = (M + TILE_M - 1) / TILE_M * TILE_M;
    return (size_t) rows * gi.K + (size_t) rows * (gi.K / gi.ag) * sizeof(float);
}

void amx_q8_0_grouped_mul_mat(const struct ggml_compute_params * params, const amx_q8_0_grouped_info & gi,
                              const struct ggml_tensor * src0, const struct ggml_tensor * src1,
                              struct ggml_tensor * dst, float * out_data, const float * bias) {
    const int K = gi.K;
    const int N = gi.N;
    // the weight is one matrix; activation rows of every batch are rows of one product
    GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && src1->ne[0] == K);
    GGML_ASSERT(dst->nb[0] == sizeof(float) && dst->nb[1] == (size_t) N * sizeof(float) && ggml_is_contiguous(dst));
    const int64_t M = ggml_nrows(src1);
    GGML_ASSERT(M == ggml_nrows(dst) && M <= INT32_MAX);
    if (params->wsize < amx_q8_0_grouped_wsize(gi, M)) {
        GGML_ABORT("insufficient work space size");
    }
    const group_scales gs(gi);
    const int          KS   = K / 64;
    const int          MT   = (int) ((M + TILE_M - 1) / TILE_M);
    int8_t *           A    = (int8_t *) params->wdata;
    float *            as   = (float *) (A + (size_t) MT * TILE_M * K);
    const float *      x    = (const float *) src1->data;
    // Passes of one row, and of up to four rows on four or more threads, run on VNNI (it streams the weights
    // faster); longer ones on the 32-column panels, which for 2 to 16 rows are faster than VNNI on fewer
    // threads and than one A tile against four weight tiles. GGML_AMX_GROUPED_VNNI_ROWS (0 to
    // AMX_GROUPED_VNNI_ROWS) overrides the limit. Every path gives the same results.
    static const int   vnni_rows_env = [] {
        const char * v = getenv("GGML_AMX_GROUPED_VNNI_ROWS");
        return v != nullptr ? std::clamp(atoi(v), 0, AMX_GROUPED_VNNI_ROWS) : -1;
    }();
    const int  vnni_rows = vnni_rows_env >= 0 ? vnni_rows_env : params->nth >= 4 ? 4 : 1;
    const bool vnni      = M <= vnni_rows;

    // VNNI passes stream the weights from memory: hand their column blocks out dynamically
    if (vnni && params->ith == 0) {
        ggml_threadpool_chunk_set(params->threadpool, params->nth);
    }
    if (vnni) {
        parallel_for_ggml(params, (int) M, [&](int begin, int end) {
            for (int m = begin; m < end; ++m) {
                quantize_row<true>(x + (size_t) m * K, A + (size_t) m * K, as + (size_t) m * gs.NGa, K, gi.ag);
            }
        });
    } else parallel_for_ggml(params, MT * TILE_M, [&](int begin, int end) {
        for (int m = begin; m < end; ++m) {
            int8_t * a = A + (size_t) (m / TILE_M) * KS * G_TILE + (m % TILE_M) * 64;
            if (m < M) {
                quantize_row<false>(x + (size_t) m * K, a, as + (size_t) m * gs.NGa, K, gi.ag);
            } else {
                for (int ks = 0; ks < KS; ++ks) {
                    memset(a + (size_t) ks * G_TILE, 0, 64);
                }
                for (int g = 0; g < gs.NGa; ++g) {
                    as[(size_t) m * gs.NGa + g] = 0.0f;
                }
            }
        }
    });
    ggml_barrier(params->threadpool);

    const int8_t * W  = (const int8_t *) src0->data;
    const float *  Ws = (const float *) (W + (size_t) N * K);
    if (vnni) {
        const int32_t * Wsum = (const int32_t *) (Ws + (size_t) N * gs.NGw);
        const int       NT   = N / TILE_N;
        const int       NB   = (NT + 3) / 4;
        for (int i = params->ith; i < NB; i = ggml_threadpool_chunk_add(params->threadpool, 1)) {
            grouped_vnni_dispatch<AMX_GROUPED_VNNI_ROWS>(gi, gs, (int) M, std::min(4, NT - 4 * i), W, Ws, Wsum,
                                                         (const uint8_t *) A, as, 4 * i, out_data + (size_t) i * 64, N,
                                                         bias != nullptr ? bias + (size_t) i * 64 : nullptr);
        }
        return;
    }
    parallel_for_ggml(params, N / (2 * TILE_N), [&](int begin, int end) {
        if (begin < end) {
            grouped_tile_config();
        }
        for (int nb = begin; nb < end; ++nb) {
            grouped_panel(gi, gs, W, Ws, A, as, (int) M, nb, out_data + (size_t) nb * 2 * TILE_N, N,
                          bias != nullptr ? bias + (size_t) nb * 2 * TILE_N : nullptr);
        }
    });
}

#endif  // defined(__AMX_INT8__) && defined(__AVX512VNNI__)
