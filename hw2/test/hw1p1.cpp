#include <cmath>
#include <conv_cpu.h>
#include <conv_gpu.h>
#include <gtest/gtest.h>
#include <iostream>

TEST(HW1P1, RUN) {
    constexpr float max_tol = 1e-7f;

    float* fm = new float[8 * 64 * 128 * 128];
    float* kern = new float[128 * 64 * 3 * 3];
    // random init them
    init_tensor(fm, 8, 64, 128, 128);
    init_tensor(kern, 128, 64, 3, 3);
    // compute by cpu
    float* cpu_res = new float[8 * 128 * 126 * 126];
    conv_cpu(cpu_res, fm, kern, 8, 64, 128, 128, 128, 3, 3);
    // compute by gpu
    float* gpu_res = new float[8 * 128 * 126 * 126];
    conv_gpu(gpu_res, fm, kern, 8, 64, 128, 128, 128, 3, 3);
    // compute max error
    float max_error = 0.0f;
    for (int i = 0; i < 8 * 128 * 126 * 126; i++) {
        max_error = fmax(max_error, fabs(cpu_res[i] - gpu_res[i]));
    }
    // std::cout << cpu_res[100] << ' ' << gpu_res[100] << '\n';
    EXPECT_TRUE(fabs(max_error) < max_tol);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}