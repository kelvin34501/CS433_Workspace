#include <benchmark/benchmark.h>
#include <iostream>
#include <random>
#include <vector>

#include "hw1p1.h"

template <typename T> auto KEEP_UNUSED(const T&) -> void {}

static auto benchmark_generic_sum(benchmark::State& state) -> void {
    // setup things
    std::random_device rnd_dev;
    std::mt19937 mersenne_engine{rnd_dev()};
    std::uniform_int_distribution<int32_t> dist{0, 1};
    auto gen = [&dist, &mersenne_engine] { return dist(mersenne_engine); };
    std::vector<int32_t> vec(10000000);
    std::generate(begin(vec), end(vec), gen);
    auto display = 0;

    // start testing
    for (auto _ : state) {
        display = sum_iter(std::begin(vec), std::end(vec));
    }

    volatile auto keep_unused = 0;
    keep_unused = display;
    KEEP_UNUSED(keep_unused);
}

static auto benchmark_generic_sum_omp(benchmark::State& state) -> void {
    // setup things
    std::random_device rnd_dev;
    std::mt19937 mersenne_engine{rnd_dev()};
    std::uniform_int_distribution<int32_t> dist{0, 1};
    auto gen = [&dist, &mersenne_engine] { return dist(mersenne_engine); };
    std::vector<int32_t> vec(10000000);
    std::generate(begin(vec), end(vec), gen);
    auto display = 0;

    auto thread_count = state.range(0);

    // start testing
    for (auto _ : state) {
        display = sum_iter_par(std::begin(vec), std::end(vec), thread_count);
    }

    volatile auto keep_unused = 0;
    keep_unused = display;
    KEEP_UNUSED(keep_unused);
}

static auto benchmark_generic_sum_omp_lock(benchmark::State& state) -> void {
    // setup things
    std::random_device rnd_dev;
    std::mt19937 mersenne_engine{rnd_dev()};
    std::uniform_int_distribution<int32_t> dist{0, 1};
    auto gen = [&dist, &mersenne_engine] { return dist(mersenne_engine); };
    std::vector<int32_t> vec(10000000);
    std::generate(begin(vec), end(vec), gen);
    auto display = 0;

    auto thread_count = state.range(0);

    // start testing
    for (auto _ : state) {
        display =
            sum_iter_par_lock(std::begin(vec), std::end(vec), thread_count);
    }

    volatile auto keep_unused = 0;
    keep_unused = display;
    KEEP_UNUSED(keep_unused);
}

static auto benchmark_generic_sum_omp_red(benchmark::State& state) -> void {
    // setup things
    std::random_device rnd_dev;
    std::mt19937 mersenne_engine{rnd_dev()};
    std::uniform_int_distribution<int32_t> dist{0, 1};
    auto gen = [&dist, &mersenne_engine] { return dist(mersenne_engine); };
    std::vector<int32_t> vec(10000000);
    std::generate(begin(vec), end(vec), gen);
    auto display = 0;

    auto thread_count = state.range(0);

    // start testing
    for (auto _ : state) {
        display =
            sum_iter_par_red(std::begin(vec), std::end(vec), thread_count);
    }

    volatile auto keep_unused = 0;
    keep_unused = display;
    KEEP_UNUSED(keep_unused);
}

// register function as benchmark
BENCHMARK(benchmark_generic_sum);
BENCHMARK(benchmark_generic_sum_omp)
    ->Arg(1)
    ->Arg(2)
    ->Arg(3)
    ->Arg(4)
    ->Arg(5)
    ->Arg(6)
    ->Arg(7)
    ->Arg(8);
BENCHMARK(benchmark_generic_sum);
BENCHMARK(benchmark_generic_sum_omp_lock)
    ->Arg(1)
    ->Arg(2)
    ->Arg(3)
    ->Arg(4)
    ->Arg(5)
    ->Arg(6)
    ->Arg(7)
    ->Arg(8);
BENCHMARK(benchmark_generic_sum);
BENCHMARK(benchmark_generic_sum_omp_red)
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