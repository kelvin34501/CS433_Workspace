#include <cmath>
#include <conv_cpu.h>
#include <conv_gpu.h>
#include <gtest/gtest.h>
#include <iostream>

TEST(HW1P2, RUN) {
    constexpr float max_tol = 1e-7f;

    float* fm = new float[8 * 64 * 128 * 128];
    float* kern = new float[128 * 64 * 3 * 3];
    // random init them
    init_tensor(fm, 8, 64, 128, 128);
    init_tensor(kern, 128, 64, 3, 3);
    // compute by cpu
    float* gpu_res = new float[8 * 128 * 126 * 126];
    conv_gpu(gpu_res, fm, kern, 8, 64, 128, 128, 128, 3, 3);
    // compute by gpu
    float* gpu_res_1 = new float[8 * 128 * 126 * 126];
    conv_gpu_2(gpu_res_1, fm, kern, 8, 64, 128, 128, 128, 3, 3);

    // compute max error
    float max_error = 0.0f;
    for (int i = 0; i < 8 * 128 * 126 * 126; i++) {
        max_error = fmax(max_error, fabs(gpu_res[i] - gpu_res_1[i]));
    }
    EXPECT_TRUE(fabs(max_error) < max_tol);
}

TEST(HW1P2, RUN_2) {
    constexpr float max_tol = 1e-7f;

    float* fm = new float[8 * 64 * 128 * 128];
    float* kern = new float[128 * 64 * 3 * 3];
    // random init them
    init_tensor(fm, 8, 64, 128, 128);
    init_tensor(kern, 128, 64, 3, 3);
    // compute by cpu
    float* gpu_res = new float[8 * 128 * 126 * 126];
    conv_gpu(gpu_res, fm, kern, 8, 64, 128, 128, 128, 3, 3);
    // compute by gpu
    float* gpu_res_1 = new float[8 * 128 * 126 * 126];
    conv_gpu_3(gpu_res_1, fm, kern, 8, 64, 128, 128, 128, 3, 3);

    // compute max error
    float max_error = 0.0f;
    int acc = 0;
    for (int i = 0; i < 8 * 128 * 126 * 126; i++) {
        if (fabs(gpu_res[i] - gpu_res_1[i]) > max_tol) {
            std::cout << i << ' ' << gpu_res[i] << ' ' << gpu_res_1[i] << '\n';
            acc++;
            if (acc > 20) {
                exit(0);
            }
        }
        max_error = fmax(max_error, fabs(gpu_res[i] - gpu_res_1[i]));
    }
    EXPECT_TRUE(fabs(max_error) < max_tol);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}