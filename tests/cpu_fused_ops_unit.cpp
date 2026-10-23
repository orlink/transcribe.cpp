// cpu_fused_ops_unit.cpp - CPU fusions that must give the bits of the ops they replace.
//
// Cases covered:
//   - SIGMOID + MUL (a gated linear unit, fused by the CPU backend) and a lone SIGMOID: equal to
//     g * (1 / (1 + expf(-x))) and 1 / (1 + expf(-x)) bit for bit with glibc's expf (within 1 ulp with other
//     libms), for ordinary values, 0, -0, +-inf, nan, subnormals and the overflow / underflow ends, on rows whose
//     length is not a multiple of the vector width.
//   - The direct conv kernels of ggml-cpu/fused-ops.h (custom ops from the CPU backend's proc address): equal
//     bit for bit to the generic ops they replace, for inputs spanning six orders of magnitude, several shapes
//     and paddings. Skipped where the backend returns no kernel.
//   - Layer norm fused by the CPU backend (NORM + MUL + ADD) against the unfused nodes, for 1-133 rows.

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

using ConvOpGetter = ggml_custom_op_t (*)(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, int);

ConvOpGetter conv_op_getter(ggml_backend_t cpu, const char * name) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu));
    return reinterpret_cast<ConvOpGetter>(ggml_backend_reg_get_proc_address(reg, name));
}

// values of both signs spanning six orders of magnitude, so that a different summation order shows
std::vector<float> wide_values(std::mt19937 & rng, size_t n) {
    std::normal_distribution<float>       nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ue(-3.0f, 3.0f);
    std::vector<float>                    v(n);
    for (float & x : v) {
        x = nd(rng) * std::pow(10.0f, ue(rng));
    }
    return v;
}

int64_t count_differ(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.empty() || a.size() != b.size()) {
        return -1;
    }
    int64_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        n += !same_bits(a[i], b[i]);
    }
    return n;
}

int64_t conv_out(int64_t n, int pad) {
    return (n + 2 * pad - 3) / 2 + 1;
}

// conv0 of a conformer pre-encode: ggml_conv_2d (1 input channel, 3x3, stride 2) + bias + ReLU
void test_conv_2d_s2_c1(ggml_backend_t cpu) {
    const ConvOpGetter get = conv_op_getter(cpu, "ggml_backend_cpu_conv_2d_s2_c1_bias_relu_op");
    if (get == nullptr) {
        std::printf("conv_2d_s2_c1: no kernel, skipped\n");
        return;
    }
    std::mt19937  rng(11);
    const int64_t C = 256;
    for (const int64_t W : { 128, 80 }) {
        for (const int64_t H : { 36, 101, 530 }) {
            for (const int pad : { 1, 2 }) {
                const Input x{
                    { W, H, 1, 1 },
                    wide_values(rng, W * H)
                };
                const Input w{
                    { 3, 3, 1, C },
                    wide_values(rng, 9 * C)
                };
                const Input              b{ { C }, wide_values(rng, C) };
                const std::vector<float> ref =
                    run_graph(cpu, { x, w, b }, [&](ggml_context * ctx, std::vector<ggml_tensor *> & t) {
                        ggml_tensor * y = ggml_conv_2d(ctx, t[1], t[0], 2, 2, pad, pad, 1, 1);
                        y               = ggml_add(ctx, y, ggml_reshape_4d(ctx, t[2], 1, 1, C, 1));
                        return ggml_relu(ctx, y);
                    });
                ggml_custom_op_t         op = nullptr;
                const std::vector<float> got =
                    run_graph(cpu, { x, w, b }, [&](ggml_context * ctx, std::vector<ggml_tensor *> & t) {
                        op = get(t[0], t[1], t[2], pad);
                        if (op == nullptr) {
                            return ggml_relu(ctx, ggml_conv_2d(ctx, t[1], t[0], 2, 2, pad, pad, 1, 1));
                        }
                        ggml_tensor * args[3] = { t[0], t[1], t[2] };
                        return ggml_custom_4d(ctx, GGML_TYPE_F32, conv_out(W, pad), conv_out(H, pad), C, 1, args, 3, op,
                                              GGML_N_TASKS_MAX, reinterpret_cast<void *>(static_cast<intptr_t>(pad)));
                    });
                if (op == nullptr) {
                    continue;  // a shape the kernel does not reproduce
                }
                const int64_t bad = count_differ(ref, got);
                if (bad != 0) {
                    std::fprintf(stderr, "conv_2d_s2_c1 W=%lld H=%lld pad=%d: %lld differ\n", (long long) W,
                                 (long long) H, pad, (long long) bad);
                }
                CHECK(bad == 0);
            }
        }
    }
}

// conv2 / conv5 of a conformer pre-encode: f32 im2col with a [9 x 1] matmul per channel (as conformer's
// conv_2d_dw_f32 builds it) + bias
void test_conv_2d_dw_s2(ggml_backend_t cpu) {
    const ConvOpGetter get = conv_op_getter(cpu, "ggml_backend_cpu_conv_2d_dw_s2_bias_op");
    if (get == nullptr) {
        std::printf("conv_2d_dw_s2: no kernel, skipped\n");
        return;
    }
    std::mt19937  rng(13);
    const int64_t C = 64;
    for (const int64_t W : { 64, 32, 20 }) {
        for (const int64_t H : { 17, 124, 265 }) {
            for (const int pad : { 1, 2 }) {
                const Input x{
                    { W, H, C, 1 },
                    wide_values(rng, W * H * C)
                };
                const Input w{
                    { 3, 3, 1, C },
                    wide_values(rng, 9 * C)
                };
                const Input              b{ { C }, wide_values(rng, C) };
                const std::vector<float> ref =
                    run_graph(cpu, { x, w, b }, [&](ggml_context * ctx, std::vector<ggml_tensor *> & t) {
                        ggml_tensor * k  = ggml_reshape_4d(ctx, t[1], 3, 3, 1, C);
                        ggml_tensor * d  = ggml_reshape_4d(ctx, t[0], W, H, 1, C);
                        ggml_tensor * ic = ggml_im2col(ctx, k, d, 2, 2, pad, pad, 1, 1, true, GGML_TYPE_F32);
                        ggml_tensor * nb = ggml_reshape_4d(ctx, ic, ic->ne[0], ic->ne[2] * ic->ne[1], C, 1);
                        ggml_tensor * y  = ggml_mul_mat(ctx, ggml_reshape_4d(ctx, k, 9, 1, C, 1), nb);
                        y                = ggml_reshape_4d(ctx, y, ic->ne[1], ic->ne[2], C, 1);
                        return ggml_add(ctx, y, ggml_reshape_4d(ctx, t[2], 1, 1, C, 1));
                    });
                ggml_custom_op_t         op = nullptr;
                const std::vector<float> got =
                    run_graph(cpu, { x, w, b }, [&](ggml_context * ctx, std::vector<ggml_tensor *> & t) {
                        op = get(t[0], t[1], t[2], pad);
                        if (op == nullptr) {
                            return ggml_cont(ctx, t[0]);
                        }
                        ggml_tensor * args[3] = { t[0], t[1], t[2] };
                        return ggml_custom_4d(ctx, GGML_TYPE_F32, conv_out(W, pad), conv_out(H, pad), C, 1, args, 3, op,
                                              GGML_N_TASKS_MAX, reinterpret_cast<void *>(static_cast<intptr_t>(pad)));
                    });
                CHECK(op != nullptr);
                const int64_t bad = count_differ(ref, got);
                if (bad != 0) {
                    std::fprintf(stderr, "conv_2d_dw_s2 W=%lld H=%lld pad=%d: %lld differ\n", (long long) W,
                                 (long long) H, pad, (long long) bad);
                }
                CHECK(bad == 0);
            }
        }
    }
}

// conv3 / conv6 of a conformer pre-encode: ggml_conv_2d with a 1x1 kernel + bias + ReLU
void test_conv_2d_pw(ggml_backend_t cpu) {
    const ConvOpGetter get = conv_op_getter(cpu, "ggml_backend_cpu_conv_2d_pw_bias_relu_op");
    if (get == nullptr) {
        std::printf("conv_2d_pw: no kernel, skipped\n");
        return;
    }
    std::mt19937  rng(17);
    const int64_t C = 256;
    for (const int64_t W : { 32, 16, 20 }) {
        for (const int64_t H : { 9, 62, 133, 300 }) {
            const Input x{
                { W, H, C, 1 },
                wide_values(rng, W * H * C)
            };
            const Input w{
                { 1, 1, C, C },
                wide_values(rng, C * C)
            };
            const Input              b{ { C }, wide_values(rng, C) };
            const std::vector<float> ref =
                run_graph(cpu, { x, w, b }, [&](ggml_context * ctx, std::vector<ggml_tensor *> & t) {
                    ggml_tensor * y = ggml_conv_2d(ctx, t[1], t[0], 1, 1, 0, 0, 1, 1);
                    y               = ggml_add(ctx, y, ggml_reshape_4d(ctx, t[2], 1, 1, C, 1));
                    return ggml_relu(ctx, y);
                });
            ggml_custom_op_t         op = nullptr;
            const std::vector<float> got =
                run_graph(cpu, { x, w, b }, [&](ggml_context * ctx, std::vector<ggml_tensor *> & t) {
                    op = get(t[0], t[1], t[2], 0);
                    if (op == nullptr) {
                        return ggml_cont(ctx, t[0]);
                    }
                    ggml_tensor * args[3] = { t[0], t[1], t[2] };
                    return ggml_custom_4d(ctx, GGML_TYPE_F32, W, H, C, 1, args, 3, op, GGML_N_TASKS_MAX, nullptr);
                });
            CHECK(op != nullptr || (W * H) % 4 != 0);
            if (op == nullptr) {
                continue;
            }
            const int64_t bad = count_differ(ref, got);
            if (bad != 0) {
                std::fprintf(stderr, "conv_2d_pw W=%lld H=%lld: %lld differ\n", (long long) W, (long long) H,
                             (long long) bad);
            }
            CHECK(bad == 0);
        }
    }
}

// the conv module of a conformer block after its gate: transpose, ggml_conv_2d_dw_direct along time (k = 9),
// bias, batch norm (multiply, add), SiLU, transpose back
void test_conv_1d_dw_bn_silu(ggml_backend_t cpu) {
    using Getter           = ggml_custom_op_t (*)(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *,
                                                  const ggml_tensor *, const ggml_tensor *, int);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu));
    const Getter       get =
        reinterpret_cast<Getter>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_conv_1d_dw_bn_silu_op"));
    if (get == nullptr) {
        std::printf("conv_1d_dw_bn_silu: no kernel, skipped\n");
        return;
    }
    std::mt19937                          rng(19);
    std::uniform_real_distribution<float> us(0.5f, 2.0f);
    for (const int64_t C : { 64, 1024 }) {
        for (const int64_t T : { 5, 13, 64, 124, 133 }) {
            for (const bool with_bias : { true, false }) {
                const Input x{
                    { C, T },
                    wide_values(rng, C * T)
                };
                const Input w{
                    { 9, 1, C },
                    wide_values(rng, 9 * C)
                };
                Input scale{ { C }, std::vector<float>(C) };
                for (float & v : scale.data) {
                    v = us(rng);
                }
                const Input              shift{ { C }, wide_values(rng, C) };
                const Input              bias{ { C }, wide_values(rng, C) };
                const std::vector<float> ref = run_graph(
                    cpu, { x, w, scale, shift, bias }, [&](ggml_context * ctx, std::vector<ggml_tensor *> & t) {
                        ggml_tensor * y = ggml_cont(ctx, ggml_permute(ctx, t[0], 1, 0, 2, 3));
                        ggml_tensor * k = ggml_reshape_4d(ctx, t[1], 9, 1, 1, C);
                        y = ggml_conv_2d_dw_direct(ctx, k, ggml_reshape_4d(ctx, y, T, 1, C, 1), 1, 1, 4, 0, 1, 1);
                        y = ggml_reshape_2d(ctx, y, T, C);
                        if (with_bias) {
                            y = ggml_add(ctx, y, ggml_reshape_2d(ctx, t[4], 1, C));
                        }
                        y = ggml_mul(ctx, y, ggml_reshape_4d(ctx, t[2], 1, C, 1, 1));
                        y = ggml_add(ctx, y, ggml_reshape_4d(ctx, t[3], 1, C, 1, 1));
                        y = ggml_silu(ctx, y);
                        return ggml_cont(ctx, ggml_permute(ctx, y, 1, 0, 2, 3));
                    });
                ggml_custom_op_t         op  = nullptr;
                const std::vector<float> got = run_graph(
                    cpu, { x, w, scale, shift, bias }, [&](ggml_context * ctx, std::vector<ggml_tensor *> & t) {
                        op = get(t[0], t[1], with_bias ? t[4] : nullptr, t[2], t[3], 4);
                        if (op == nullptr) {
                            return ggml_cont(ctx, t[0]);
                        }
                        ggml_tensor * args[5] = { t[0], t[1], t[2], t[3], t[4] };
                        return ggml_custom_4d(ctx, GGML_TYPE_F32, C, T, 1, 1, args, with_bias ? 5 : 4, op,
                                              GGML_N_TASKS_MAX, reinterpret_cast<void *>(static_cast<intptr_t>(4)));
                    });
                CHECK(op != nullptr);
                const int64_t bad = count_differ(ref, got);
                if (bad != 0) {
                    std::fprintf(stderr, "conv_1d_dw_bn_silu C=%lld T=%lld bias=%d: %lld differ\n", (long long) C,
                                 (long long) T, (int) with_bias, (long long) bad);
                }
                CHECK(bad == 0);
            }
        }
    }
}

// layer norm: NORM + MUL + ADD fused by the CPU backend (row sums 16 rows at a time) against the same nodes
// unfused (the NORM marked as an output, which keeps the backend from fusing it)
void test_norm_mul_add(ggml_backend_t cpu) {
    std::mt19937 rng(23);
    for (const int64_t n : { 1024, 1003 }) {
        for (const int64_t rows : { 1, 5, 31, 64, 124, 133 }) {
            const Input x{
                { n, rows },
                wide_values(rng, n * rows)
            };
            const Input        w{ { n }, wide_values(rng, n) };
            const Input        b{ { n }, wide_values(rng, n) };
            std::vector<float> res[2];
            for (int fused = 0; fused < 2; ++fused) {
                res[fused] = run_graph(cpu, { x, w, b }, [&](ggml_context * ctx, std::vector<ggml_tensor *> & t) {
                    ggml_tensor * y = ggml_norm(ctx, t[0], 1e-5f);
                    if (!fused) {
                        ggml_set_output(y);
                    }
                    return ggml_add(ctx, ggml_mul(ctx, y, t[1]), t[2]);
                });
            }
            const int64_t bad = count_differ(res[0], res[1]);
            if (bad != 0) {
                std::fprintf(stderr, "norm_mul_add n=%lld rows=%lld: %lld differ\n", (long long) n, (long long) rows,
                             (long long) bad);
            }
            CHECK(bad == 0);
        }
    }
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
    test_conv_2d_s2_c1(cpu);
    test_conv_2d_dw_s2(cpu);
    test_conv_2d_pw(cpu);
    test_conv_1d_dw_bn_silu(cpu);
    test_norm_mul_add(cpu);

    ggml_backend_free(cpu);
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("cpu_fused_ops_unit: ok\n");
    return EXIT_SUCCESS;
}
