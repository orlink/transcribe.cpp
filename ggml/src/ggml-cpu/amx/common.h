#pragma once

#include "ggml.h"
#include "ggml-cpu-impl.h"

#include <algorithm>
#include <memory>
#include <type_traits>

#if defined(GGML_USE_OPENMP)
#include <omp.h>
#else
#include <thread>
#endif

#define TILE_M 16
#define TILE_N 16
#define TILE_K 32
#define VNNI_BLK 4

#define AMX_BLK_SIZE 32

#define TMM0 0
#define TMM1 1
#define TMM2 2
#define TMM3 3
#define TMM4 4
#define TMM5 5
#define TMM6 6
#define TMM7 7

// parallel routines
template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline T div_up(T x, T y) { return (x + y - 1) / y; }

template <typename T>
inline void balance211(T n, T nth, T ith, T& n_start, T& n_end) {
#if 0
    // onednn partition pattern
    T& n_my = n_end;
    if (nth <= 1 || n == 0) {
        n_start = 0;
        n_my = n;
    } else {
        T n1 = div_up(n, nth);
        T n2 = n1 - 1;
        T T1 = n - n2 * nth;
        n_my = ith < T1 ? n1 : n2;
        n_start = ith <= T1 ? ith*n1 : T1 * n1 + (ith - T1) * n2;
    }
    n_end += n_start;
#else
    // pytorch aten partition pattern
    T n_my = div_up(n, nth);
    n_start = ith * n_my;
    n_end = std::min(n_start + n_my, n);
#endif
}

template <typename func_t>
inline void parallel_for(int n, const func_t & f) {
    if (n <= 0) {
        return;
    }
#if defined(GGML_USE_OPENMP)
    #pragma omp parallel
    {
        int nth = omp_get_num_threads();
        int ith = omp_get_thread_num();
        int tbegin, tend;
        balance211(n, nth, ith, tbegin, tend);
        f(tbegin, tend);
    }
#else
    int nth = std::thread::hardware_concurrency();
    if (nth <= 1) {
        f(0, n);
        return;
    }
    if (nth > n) {
        nth = n;
    }
    std::vector<std::thread> threads;
    threads.reserve(nth);
    for (int ith = 0; ith < nth; ++ith) {
        threads.emplace_back([&f, n, ith, nth] {
            int tbegin, tend;
            balance211(n, nth, ith, tbegin, tend);
            f(tbegin, tend);
        });
    }
    for (auto & t : threads) {
        t.join();
    }
#endif
}

template <typename func_t>
inline void parallel_for_ggml(const ggml_compute_params * params, int n, const func_t & f) {
    int tbegin, tend;
    balance211(n, params->nth, params->ith, tbegin, tend);
    f(tbegin, tend);
}

// define amx tile config data structure
struct tile_config_t{
    uint8_t palette_id = 0;
    uint8_t start_row = 0;
    uint8_t reserved_0[14] = {0};
    uint16_t colsb[16] = {0};
    uint8_t rows[16] = {0};
};

// the tile configuration this thread has loaded: the kernels of mmq.cpp and those of mmq-grouped.cpp use
// different ones, so each loads its own when the other's is active
enum { AMX_TILE_CONFIG_NONE = 0, AMX_TILE_CONFIG_Q = 1, AMX_TILE_CONFIG_GROUPED = 2 };
inline thread_local int ggml_amx_tile_config_id = AMX_TILE_CONFIG_NONE;

// quantized types that have AMX support
inline bool qtype_has_amx_kernels(const enum ggml_type type) {
    // TODO: fix padding for vnni format
    return (type == GGML_TYPE_Q4_0) ||
        (type == GGML_TYPE_Q4_1) ||
        (type == GGML_TYPE_Q8_0) ||
        (type == GGML_TYPE_Q4_K) ||
        (type == GGML_TYPE_Q5_K) ||
        (type == GGML_TYPE_Q6_K) ||
        (type == GGML_TYPE_IQ4_XS);
}
