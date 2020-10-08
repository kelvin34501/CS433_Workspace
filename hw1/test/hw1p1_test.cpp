#include <gtest/gtest.h>
#include <vector>

#include "hw1p1.h"

TEST(HW1P1, SERIAL) {
    std::vector<int32_t> vec{};
    for (int32_t i = 1; i <= 100; ++i) {
        vec.push_back(i);
    }
    EXPECT_EQ(sum_iter(std::begin(vec), std::end(vec)), 5050);

    vec.clear();
    for (int32_t i = 200; i >= 100; --i) {
        vec.push_back(i);
    }
    EXPECT_EQ(sum_iter(std::begin(vec), std::end(vec)), 15150);
}

TEST(HW1P1, PAR) {
    std::vector<int32_t> vec{};
    for (int32_t i = 1; i <= 100; ++i) {
        vec.push_back(i);
    }
    EXPECT_EQ(sum_iter_par(std::begin(vec), std::end(vec), 4), 5050);

    vec.clear();
    for (int32_t i = 200; i >= 100; --i) {
        vec.push_back(i);
    }
    EXPECT_EQ(sum_iter_par(std::begin(vec), std::end(vec), 4), 15150);
}

TEST(HW1P1, PAR_LOCK) {
    std::vector<int32_t> vec{};
    for (int32_t i = 1; i <= 100; ++i) {
        vec.push_back(i);
    }
    EXPECT_EQ(sum_iter_par_lock(std::begin(vec), std::end(vec), 4), 5050);

    vec.clear();
    for (int32_t i = 200; i >= 100; --i) {
        vec.push_back(i);
    }
    EXPECT_EQ(sum_iter_par_lock(std::begin(vec), std::end(vec), 4), 15150);
}

TEST(HW1P1, PAR_RED) {
    std::vector<int32_t> vec{};
    for (int32_t i = 1; i <= 100; ++i) {
        vec.push_back(i);
    }
    EXPECT_EQ(sum_iter_par_red(std::begin(vec), std::end(vec), 4), 5050);

    vec.clear();
    for (int32_t i = 200; i >= 100; --i) {
        vec.push_back(i);
    }
    EXPECT_EQ(sum_iter_par_red(std::begin(vec), std::end(vec), 4), 15150);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}