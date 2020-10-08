#include <array>
#include <gtest/gtest.h>

#include "hw1p2.h"

using namespace HW1P2;

TEST(HW1P2, SERIAL) {
    std::array<float, 12> arr_A{1.0, 2.0, 3.0, 4.0,  5.0,  6.0,
                                7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    std::array<float, 20> arr_B{1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,
                                8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0,
                                15.0, 16.0, 17.0, 18.0, 19.0, 20.0};
    std::array<float, 15> arr_C{110., 120., 130., 140., 150., 246., 272., 298.,
                                324., 350., 382., 424., 466., 508., 550.};
    auto A = mat<float, 3, 4>(arr_A);
    auto B = mat<float, 4, 5>(arr_B);
    auto C = mat<float, 3, 5>(arr_C);
    auto p_C_ = matmul(A, B);

    EXPECT_EQ(*p_C_, C);
}

TEST(HW1P2, SERIAL_ALT) {
    std::array<float, 12> arr_A{1.0, 2.0, 3.0, 4.0,  5.0,  6.0,
                                7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    std::array<float, 20> arr_B{1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,
                                8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0,
                                15.0, 16.0, 17.0, 18.0, 19.0, 20.0};
    std::array<float, 15> arr_C{110., 120., 130., 140., 150., 246., 272., 298.,
                                324., 350., 382., 424., 466., 508., 550.};
    auto A = mat<float, 3, 4>(arr_A);
    auto B = mat<float, 4, 5>(arr_B);
    auto B_alt = mat_alt(B);
    auto C = mat<float, 3, 5>(arr_C);
    auto p_C_ = matmul(A, B);

    EXPECT_EQ(*p_C_, C);
}

TEST(HW1P2, PAR) {
    std::array<float, 12> arr_A{1.0, 2.0, 3.0, 4.0,  5.0,  6.0,
                                7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    std::array<float, 20> arr_B{1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,
                                8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0,
                                15.0, 16.0, 17.0, 18.0, 19.0, 20.0};
    std::array<float, 15> arr_C{110., 120., 130., 140., 150., 246., 272., 298.,
                                324., 350., 382., 424., 466., 508., 550.};
    auto A = mat<float, 3, 4>(arr_A);
    auto B = mat<float, 4, 5>(arr_B);
    auto C = mat<float, 3, 5>(arr_C);
    auto p_C_ = matmul_par(A, B, 4);

    EXPECT_EQ(*p_C_, C);
}

TEST(HW1P2, PAR_ALT) {
    std::array<float, 12> arr_A{1.0, 2.0, 3.0, 4.0,  5.0,  6.0,
                                7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    std::array<float, 20> arr_B{1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,
                                8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0,
                                15.0, 16.0, 17.0, 18.0, 19.0, 20.0};
    std::array<float, 15> arr_C{110., 120., 130., 140., 150., 246., 272., 298.,
                                324., 350., 382., 424., 466., 508., 550.};
    auto A = mat<float, 3, 4>(arr_A);
    auto B = mat<float, 4, 5>(arr_B);
    auto B_alt = mat_alt(B);
    auto C = mat<float, 3, 5>(arr_C);
    auto p_C_ = matmul_par(A, B_alt, 4);

    EXPECT_EQ(*p_C_, C);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}