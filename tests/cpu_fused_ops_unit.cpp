// cpu_fused_ops_unit.cpp - CPU fusions that must give the bits of the ops they replace.
//
// Cases covered:
//   - SIGMOID + MUL (a gated linear unit, fused by the CPU backend) and a lone SIGMOID: equal to
//     g * (1 / (1 + expf(-x))) and 1 / (1 + expf(-x)) bit for bit with glibc's expf (within 1 ulp with other
//     libms), for ordinary values, 0, -0, +-inf, nan, subnormals and the overflow / underflow ends, on rows whose
//     length is not a multiple of the vector width.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
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

struct Input {
    std::vector<int64_t> ne;
    std::vector<float>   data;
};

// Build a graph over f32 inputs with `build`, compute it on the CPU backend, return the output.
std::vector<float> run_graph(ggml_backend_t                                                                     cpu,
                             const std::vector<Input> &                                                         inputs,
                             const std::function<ggml_tensor *(ggml_context *, std::vector<ggml_tensor *> &)> & build) {
    ggml_init_params           params = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context *             ctx    = ggml_init(params);
    std::vector<ggml_tensor *> ts;
    for (const Input & in : inputs) {
        ts.push_back(ggml_new_tensor(ctx, GGML_TYPE_F32, static_cast<int>(in.ne.size()), in.ne.data()));
    }
    ggml_tensor * out = build(ctx, ts);
    ggml_cgraph * gf  = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    for (size_t i = 0; i < inputs.size(); ++i) {
        ggml_backend_tensor_set(ts[i], inputs[i].data.data(), 0, ggml_nbytes(ts[i]));
    }
    std::vector<float> res;
    if (ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS) {
        res.resize(ggml_nelements(out));
        ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

bool same_bits(float a, float b) {
    return std::memcmp(&a, &b, sizeof(float)) == 0 || (std::isnan(a) && std::isnan(b));
}

bool within_ulp(float a, float b) {
    if (same_bits(a, b)) {
        return true;
    }
    return std::fabs(a - b) <= std::fabs(std::nextafter(a, 2 * b) - a);
}

void test_sigmoid_glu(ggml_backend_t cpu) {
    std::mt19937                          rng(7);
    std::uniform_real_distribution<float> ud(-30.0f, 30.0f);
    const int64_t                         n    = 1000 + 7;  // not a multiple of 16
    const int64_t                         rows = 5;
    // value and gate halves, as in a conformer GLU
    Input                                 a;
    a.ne = { 2 * n, rows };
    a.data.resize(2 * n * rows);
    for (float & v : a.data) {
        v = ud(rng);
    }
    const float specials[] = { 0.0f,    -0.0f,    INFINITY, -INFINITY, NAN,   1e-45f, -1e-45f,
                               1e-38f,  88.0f,    88.72f,   88.73f,    89.0f, -87.0f, -88.5f,
                               -103.0f, -103.97f, -104.0f,  -110.0f,   17.0f, -17.0f, 0.5f };
    for (size_t i = 0; i < sizeof(specials) / sizeof(specials[0]); ++i) {
        a.data[n + i]     = specials[i];   // values in row 0
        a.data[3 * n + i] = -specials[i];  // and negated in row 1
    }
    auto value = [&](ggml_context * ctx, ggml_tensor * x) {
        return ggml_view_2d(ctx, x, n, rows, x->nb[1], n * sizeof(float));
    };
    auto gate = [&](ggml_context * ctx, ggml_tensor * x) {
        return ggml_view_2d(ctx, x, n, rows, x->nb[1], 0);
    };
    const std::vector<float> glu = run_graph(cpu, { a }, [&](ggml_context * ctx, std::vector<ggml_tensor *> & t) {
        return ggml_mul(ctx, gate(ctx, t[0]), ggml_sigmoid(ctx, value(ctx, t[0])));
    });
    const std::vector<float> sig = run_graph(cpu, { a }, [&](ggml_context * ctx, std::vector<ggml_tensor *> & t) {
        return ggml_sigmoid(ctx, ggml_cont(ctx, value(ctx, t[0])));
    });
    CHECK(glu.size() == static_cast<size_t>(n * rows) && sig.size() == glu.size());
    if (glu.size() != static_cast<size_t>(n * rows) || sig.size() != glu.size()) {
        return;
    }
    int64_t bad_glu = 0;
    int64_t bad_sig = 0;
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t i = 0; i < n; ++i) {
            const float x  = a.data[r * 2 * n + n + i];
            const float g  = a.data[r * 2 * n + i];
            const float s  = 1.f / (1.f + expf(-x));
            const float gs = g * s;
#if defined(__GLIBC__)
            bad_sig += !same_bits(sig[r * n + i], s);
            bad_glu += !same_bits(glu[r * n + i], gs);
#else
            bad_sig += !within_ulp(sig[r * n + i], s);
            bad_glu += !within_ulp(glu[r * n + i], gs);
#endif
        }
    }
    if (bad_sig != 0 || bad_glu != 0) {
        std::fprintf(stderr, "sigmoid: %lld, gated: %lld of %lld differ\n", (long long) bad_sig, (long long) bad_glu,
                     (long long) (n * rows));
    }
    CHECK(bad_sig == 0);
    CHECK(bad_glu == 0);
}

}  // namespace

int main() {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (cpu == nullptr) {
        std::fprintf(stderr, "no CPU backend\n");
        return EXIT_FAILURE;
    }
    ggml_backend_cpu_set_n_threads(cpu, 4);

    test_sigmoid_glu(cpu);

    ggml_backend_free(cpu);
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("cpu_fused_ops_unit: ok\n");
    return EXIT_SUCCESS;
}
