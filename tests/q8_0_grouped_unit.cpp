// q8_0_grouped_unit.cpp - grouped Q8_0 scales: the quantizer and, on CPUs
// with AMX, ggml's grouped layout.
//
// Cases covered:
//   - quantize_q8_0_grouped: blocks share one scale per group (or per row),
//     values are the nearest multiples of the FP16 scale.
//   - With ScopedCpuQ8_0Grouping, a grouped weight converted into the AMX
//     buffer computes ggml_mul_mat (and a fused bias add) as the int8 sums of
//     per-row / per-group quantized activations, scaled once per group:
//     checked against the same products in double for 1-40 rows, weight
//     groups of 128 / per row and activation groups of 128 / per row.
//   - Without the scope (or with scales not shared per group) the weight
//     keeps the standard Q8_0 kernels: results within Q8_0 error of float.
// The AMX cases are skipped where ggml has no AMX buffer type.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "transcribe-backend.h"
#include "transcribe-load-common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

#define REQUIRE(cond)                                                                  \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::fprintf(stderr, "SETUP FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            return EXIT_FAILURE;                                                       \
        }                                                                              \
    } while (0)

constexpr int k_block = 32;

struct Block {
    ggml_fp16_t d;
    int8_t      qs[k_block];
};

// the activation quantization of the grouped kernels: per group, d = amax / 127, q = nearest(x * (1 / d))
void quantize_act(const float * x, int K, int ag, std::vector<int> & q, std::vector<float> & d) {
    q.assign(K, 0);
    d.assign(K / ag, 0.0f);
    for (int g = 0; g < K / ag; ++g) {
        float amax = 0.0f;
        for (int k = g * ag; k < (g + 1) * ag; ++k) {
            amax = std::max(amax, std::fabs(x[k]));
        }
        const float dd = amax / 127.0f;
        const float id = dd != 0.0f ? 1.0f / dd : 0.0f;
        d[g]           = dd;
        for (int k = g * ag; k < (g + 1) * ag; ++k) {
            q[k] = static_cast<int>(std::nearbyint(x[k] * id));
        }
    }
}

ggml_backend_buffer_type_t amx_buft(ggml_backend_t cpu) {
    ggml_backend_dev_t dev       = ggml_backend_get_device(cpu);
    auto               get_extra = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts"));
    if (get_extra == nullptr) {
        return nullptr;
    }
    for (ggml_backend_buffer_type_t * b = get_extra(dev); b != nullptr && *b != nullptr; ++b) {
        if (std::strcmp(ggml_backend_buft_name(*b), "AMX") == 0) {
            return *b;
        }
    }
    return nullptr;
}

// out = x @ w^T + bias for a Q8_0 weight in the AMX buffer, computed by the CPU backend
bool run_matmul(ggml_backend_t                                cpu,
                ggml_backend_buffer_type_t                    amx,
                const transcribe::BackendPlan &               plan,
                const transcribe::load_common::Q8_0Grouping & grouping,
                const std::vector<Block> &                    wq,
                int                                           K,
                int                                           N,
                const std::vector<float> &                    x,
                int                                           M,
                const std::vector<float> &                    bias,
                std::vector<float> &                          out) {
    ggml_init_params      wp   = { 2 * ggml_tensor_overhead(), nullptr, true };
    ggml_context *        wctx = ggml_init(wp);
    ggml_tensor *         w    = ggml_new_tensor_2d(wctx, GGML_TYPE_Q8_0, K, N);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, amx);
    if (wbuf == nullptr) {
        ggml_free(wctx);
        return false;
    }
    {
        const transcribe::load_common::ScopedCpuQ8_0Grouping scope(plan, grouping);
        ggml_backend_tensor_set(w, wq.data(), 0, ggml_nbytes(w));
    }

    ggml_init_params cp   = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context *   cctx = ggml_init(cp);
    ggml_tensor *    xt   = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, K, M);
    ggml_tensor *    bt   = ggml_new_tensor_1d(cctx, GGML_TYPE_F32, N);
    ggml_tensor *    y    = ggml_add(cctx, ggml_mul_mat(cctx, w, xt), bt);
    ggml_cgraph *    gf   = ggml_new_graph(cctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t cbuf = ggml_backend_alloc_ctx_tensors(cctx, cpu);
    ggml_backend_tensor_set(xt, x.data(), 0, ggml_nbytes(xt));
    ggml_backend_tensor_set(bt, bias.data(), 0, ggml_nbytes(bt));
    const bool ok = ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS;
    out.resize(static_cast<size_t>(M) * N);
    ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
    ggml_backend_buffer_free(cbuf);
    ggml_free(cctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    return ok;
}

void test_quantizer() {
    std::mt19937                    rng(1);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int                       K = 256;
    std::vector<float>              x(2 * K);
    for (float & v : x) {
        v = nd(rng);
    }
    for (int group : { 64, 128, 0 }) {
        std::vector<Block> q(2 * K / k_block);
        transcribe::load_common::quantize_q8_0_grouped(x.data(), q.data(), 2, K, group);
        const int g = group == 0 ? K : group;
        for (int r = 0; r < 2; ++r) {
            for (int g0 = 0; g0 < K; g0 += g) {
                float amax = 0.0f;
                for (int k = g0; k < g0 + g; ++k) {
                    amax = std::max(amax, std::fabs(x[r * K + k]));
                }
                const ggml_fp16_t dh = ggml_fp32_to_fp16(amax / 127.0f);
                const float       d  = ggml_fp16_to_fp32(dh);
                for (int k = g0; k < g0 + g; ++k) {
                    const Block & b = q[(r * K + k) / k_block];
                    CHECK(b.d == dh);
                    CHECK(std::fabs(b.qs[k % k_block] * d - x[r * K + k]) <= 0.5f * d * 1.001f + 1e-7f);
                }
            }
        }
    }
}

}  // namespace

int main() {
    using namespace transcribe;
    ggml_log_set(
        [](ggml_log_level level, const char * text, void *) {
            if (level >= GGML_LOG_LEVEL_WARN) {
                std::fputs(text, stderr);
            }
        },
        nullptr);
    test_quantizer();

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    REQUIRE(cpu != nullptr);
    ggml_backend_buffer_type_t amx = amx_buft(cpu);
    if (amx == nullptr) {
        std::fprintf(stderr, "q8_0_grouped_unit: no AMX buffer type, layout cases skipped\n");
        ggml_backend_free(cpu);
        return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu)), "ggml_backend_set_n_threads"))) {
        set_threads(cpu, 3);
    }

    BackendPlan plan;
    plan.primary      = cpu;
    plan.primary_kind = BackendKind::Cpu;
    plan.scheduler_list.push_back(cpu);

    std::mt19937                    rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int                       N = 96;  // a 64-column block and a 32-column tail for short passes
    for (int K : { 256, 1024 }) {
        std::vector<float> wf(static_cast<size_t>(N) * K);
        for (float & v : wf) {
            v = nd(rng) * 0.05f;
        }
        std::vector<float> bias(N);
        for (float & v : bias) {
            v = nd(rng);
        }
        for (int wg : { 128, 0 }) {
            std::vector<Block> wq(static_cast<size_t>(N) * K / k_block);
            load_common::quantize_q8_0_grouped(wf.data(), wq.data(), N, K, wg);
            for (int ag : { 0, 128, -1 }) {
                // ag = -1: the same weight converted without the scope (standard kernels); it must not match
                // the grouped products, which shows the cases above ran the grouped kernels
                load_common::Q8_0Grouping grouping;
                grouping.weight_group = ag >= 0 ? wg : -1;
                grouping.act_group    = std::max(ag, 0);
                for (int M : { 1, 3, 16, 17, 40 }) {
                    std::vector<float> x(static_cast<size_t>(M) * K);
                    for (float & v : x) {
                        v = nd(rng);
                    }
                    std::vector<float> got;
                    REQUIRE(run_matmul(cpu, amx, plan, grouping, wq, K, N, x, M, bias, got));
                    const int A   = ag <= 0 ? K : ag;
                    const int W   = wg == 0 ? K : wg;
                    const int R   = std::min(A, W);
                    double    err = 0.0;
                    for (int m = 0; m < M; ++m) {
                        std::vector<int>   qa;
                        std::vector<float> da;
                        quantize_act(&x[static_cast<size_t>(m) * K], K, A, qa, da);
                        for (int n = 0; n < N; ++n) {
                            double ref = 0.0;
                            double mag = 0.0;
                            for (int g0 = 0; g0 < K; g0 += R) {
                                int64_t s = 0;
                                for (int k = g0; k < g0 + R; ++k) {
                                    s += static_cast<int64_t>(qa[k]) *
                                         wq[(static_cast<size_t>(n) * K + k) / k_block].qs[k % k_block];
                                }
                                const double dw = ggml_fp16_to_fp32(wq[(static_cast<size_t>(n) * K + g0) / k_block].d);
                                const double term = static_cast<double>(s) * dw * da[g0 / A];
                                ref += term;
                                mag += std::fabs(term);
                            }
                            ref += bias[n];
                            mag += std::fabs(bias[n]);
                            const double e =
                                std::fabs(got[static_cast<size_t>(m) * N + n] - ref) / std::max(mag, 1e-30);
                            err = std::max(err, e);
                        }
                    }
                    if ((ag >= 0) != (err <= 1e-5)) {
                        std::fprintf(stderr, "FAIL K=%d wg=%d ag=%d M=%d: max relative error %.3g\n", K, wg, ag, M,
                                     err);
                        ++g_failures;
                    }
                }
            }
        }

        // scales per 32 (not shared per group): the standard kernels, close to the float product
        std::vector<Block> wq(static_cast<size_t>(N) * K / k_block);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, wf.data(), wq.data(), 0, N, K, nullptr);
        load_common::Q8_0Grouping grouping;
        grouping.weight_group = 128;
        for (int M : { 1, 17 }) {
            std::vector<float> x(static_cast<size_t>(M) * K);
            for (float & v : x) {
                v = nd(rng);
            }
            std::vector<float> got;
            REQUIRE(run_matmul(cpu, amx, plan, grouping, wq, K, N, x, M, bias, got));
            double err = 0.0;
            for (int m = 0; m < M; ++m) {
                for (int n = 0; n < N; ++n) {
                    double ref = bias[n];
                    double mag = std::fabs(bias[n]);
                    for (int k = 0; k < K; ++k) {
                        const double t =
                            static_cast<double>(x[static_cast<size_t>(m) * K + k]) * wf[static_cast<size_t>(n) * K + k];
                        ref += t;
                        mag += std::fabs(t);
                    }
                    err = std::max(err, std::fabs(got[static_cast<size_t>(m) * N + n] - ref) / mag);
                }
            }
            CHECK(err < 0.02);
        }
    }

    ggml_backend_free(cpu);
    if (g_failures != 0) {
        std::fprintf(stderr, "q8_0_grouped_unit: %d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("q8_0_grouped_unit: ok\n");
    return EXIT_SUCCESS;
}
