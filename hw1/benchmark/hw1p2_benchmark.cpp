#include <array>
#include <benchmark/benchmark.h>
#include <iostream>
#include <memory>
#include <random>
#include <utility>

#include "hw1p2.h"
using namespace HW1P2;

static auto benchmark_matmul(benchmark::State& state) -> void {
    // setup things
    std::random_device rnd_dev;
    std::mt19937 mersenne_engine{rnd_dev()};
    std::uniform_real_distribution<float> dist{0, 1};
    auto gen = [&dist, &mersenne_engine] { return dist(mersenne_engine); };
    auto p_arr_A = std::make_unique<std::array<float, 120000>>();
    std::generate(std::begin(*p_arr_A), std::end(*p_arr_A), gen);
    auto p_arr_B = std::make_unique<std::array<float, 200000>>();
    std::generate(std::begin(*p_arr_B), std::end(*p_arr_B), gen);
    auto A = std::make_unique<mat<float, 300, 400>>(*p_arr_A);
    auto B = std::make_unique<mat<float, 400, 500>>(*p_arr_B);
    auto p_display = std::make_unique<mat<float, 300, 500>>();

    // start testing
    for (auto _ : state) {
        p_display = matmul(*A, *B);
    }
}

static auto benchmark_matmul_alt(benchmark::State& state) -> void {
    // setup things
    std::random_device rnd_dev;
    std::mt19937 mersenne_engine{rnd_dev()};
    std::uniform_real_distribution<float> dist{0, 1};
    auto gen = [&dist, &mersenne_engine] { return dist(mersenne_engine); };
    auto p_arr_A = std::make_unique<std::array<float, 120000>>();
    std::generate(std::begin(*p_arr_A), std::end(*p_arr_A), gen);
    auto p_arr_B = std::make_unique<std::array<float, 200000>>();
    std::generate(std::begin(*p_arr_B), std::end(*p_arr_B), gen);
    auto A = std::make_unique<mat<float, 300, 400>>(*p_arr_A);
    auto B = std::make_unique<mat<float, 400, 500>>(*p_arr_B);
    auto B_alt = std::make_unique<mat_alt<float, 400, 500>>(*B);
    auto p_display = std::make_unique<mat<float, 300, 500>>();

    // start testing
    for (auto _ : state) {
        p_display = matmul(*A, *B_alt);
    }
}

static auto benchmark_matmul_par(benchmark::State& state) -> void {
    // setup things
    std::random_device rnd_dev;
    std::mt19937 mersenne_engine{rnd_dev()};
    std::uniform_real_distribution<float> dist{0, 1};
    auto gen = [&dist, &mersenne_engine] { return dist(mersenne_engine); };
    auto p_arr_A = std::make_unique<std::array<float, 120000>>();
    std::generate(std::begin(*p_arr_A), std::end(*p_arr_A), gen);
    auto p_arr_B = std::make_unique<std::array<float, 200000>>();
    std::generate(std::begin(*p_arr_B), std::end(*p_arr_B), gen);
    auto A = std::make_unique<mat<float, 300, 400>>(*p_arr_A);
    auto B = std::make_unique<mat<float, 400, 500>>(*p_arr_B);
    auto p_display = std::make_unique<mat<float, 300, 500>>();

    auto thread_count = state.range(0);

    // start testing
    for (auto _ : state) {
        p_display = matmul_par(*A, *B, thread_count);
    }
}

static auto benchmark_matmul_par_alt(benchmark::State& state) -> void {
    // setup things
    std::random_device rnd_dev;
    std::mt19937 mersenne_engine{rnd_dev()};
    std::uniform_real_distribution<float> dist{0, 1};
    auto gen = [&dist, &mersenne_engine] { return dist(mersenne_engine); };
    auto p_arr_A = std::make_unique<std::array<float, 120000>>();
    std::generate(std::begin(*p_arr_A), std::end(*p_arr_A), gen);
    auto p_arr_B = std::make_unique<std::array<float, 200000>>();
    std::generate(std::begin(*p_arr_B), std::end(*p_arr_B), gen);
    auto A = std::make_unique<mat<float, 300, 400>>(*p_arr_A);
    auto B = std::make_unique<mat<float, 400, 500>>(*p_arr_B);
    auto B_alt = std::make_unique<mat_alt<float, 400, 500>>(*B);
    auto p_display = std::make_unique<mat<float, 300, 500>>();

    auto thread_count = state.range(0);

    // start testing
    for (auto _ : state) {
        p_display = matmul_par(*A, *B_alt, thread_count);
    }
}

// register function as benchmark
BENCHMARK(benchmark_matmul);
BENCHMARK(benchmark_matmul_par)
    ->Arg(1)
    ->Arg(2)
    ->Arg(3)
    ->Arg(4)
    ->Arg(5)
    ->Arg(6)
    ->Arg(7)
    ->Arg(8);
BENCHMARK(benchmark_matmul_alt);
BENCHMARK(benchmark_matmul_par_alt)
    ->Arg(1)
    ->Arg(2)
    ->Arg(3)
    ->Arg(4)
    ->Arg(5)
    ->Arg(6)
    ->Arg(7)
    ->Arg(8);

// run the benchmark
BENCHMARK_MAIN();