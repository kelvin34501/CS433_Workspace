#include <benchmark/benchmark.h>
#include <conv_cpu.h>
#include <conv_gpu.h>

static void benchmark_cpu(benchmark::State& state) {

    float* fm = new float[8 * 64 * 128 * 128];
    float* kern = new float[128 * 64 * 3 * 3];
    // random init them
    init_tensor(fm, 8, 64, 128, 128);
    init_tensor(kern, 128, 64, 3, 3);
    // compute by cpu
    float* cpu_res = new float[8 * 128 * 126 * 126];
    for (auto _ : state) {
        conv_cpu(cpu_res, fm, kern, 8, 64, 128, 128, 128, 3, 3);
    }
}

static void benchmark_gpu(benchmark::State& state) {
    float* fm = new float[8 * 64 * 128 * 128];
    float* kern = new float[128 * 64 * 3 * 3];
    // random init them
    init_tensor(fm, 8, 64, 128, 128);
    init_tensor(kern, 128, 64, 3, 3);
    // compute by gpu
    float* gpu_res = new float[8 * 128 * 126 * 126];
    for (auto _ : state) {
        conv_gpu(gpu_res, fm, kern, 8, 64, 128, 128, 128, 3, 3);
    }
}

static void benchmark_gpu_2(benchmark::State& state) {
    float* fm = new float[8 * 64 * 128 * 128];
    float* kern = new float[128 * 64 * 3 * 3];
    // random init them
    init_tensor(fm, 8, 64, 128, 128);
    init_tensor(kern, 128, 64, 3, 3);
    // compute by gpu
    float* gpu_res = new float[8 * 128 * 126 * 126];
    for (auto _ : state) {
        conv_gpu_2(gpu_res, fm, kern, 8, 64, 128, 128, 128, 3, 3);
    }
}

static void benchmark_gpu_3(benchmark::State& state) {
    float* fm = new float[8 * 64 * 128 * 128];
    float* kern = new float[128 * 64 * 3 * 3];
    // random init them
    init_tensor(fm, 8, 64, 128, 128);
    init_tensor(kern, 128, 64, 3, 3);
    // compute by gpu
    float* gpu_res = new float[8 * 128 * 126 * 126];
    for (auto _ : state) {
        conv_gpu_3(gpu_res, fm, kern, 8, 64, 128, 128, 128, 3, 3);
    }
}

BENCHMARK(benchmark_cpu);
BENCHMARK(benchmark_gpu);
BENCHMARK(benchmark_gpu_2);
BENCHMARK(benchmark_gpu_3);

BENCHMARK_MAIN();