#ifndef HW1P2_H
#define HW1P2_H

#include <array>
#include <cstdint>
#include <memory>
#include <utility>

namespace HW1P2 {

template <typename TypeNumber, int32_t row, int32_t col> struct mat {
    std::array<TypeNumber, row * col> data{}; // should be row major
    int32_t n_row{row};
    int32_t n_col{col};

    // most con & decon will use default
    mat() = default;
    mat(const mat<TypeNumber, row, col>& other) = default;
    mat(mat<TypeNumber, row, col>&& other) noexcept = default;
    auto operator=(const mat<TypeNumber, row, col>& other)
        -> mat<TypeNumber, row, col>& = default;
    auto operator=(mat<TypeNumber, row, col>&& other) noexcept
        -> mat<TypeNumber, row, col>& = default;
    ~mat() noexcept = default;

    // copy con & move con from std::array
    // check cppreference
    mat(const std::array<TypeNumber, row * col>& other_data)
        : data(other_data) {}
    mat(std::array<TypeNumber, row * col>&& other_data) : data(other_data) {}
};

template <typename TypeNumber, int32_t row, int32_t col> struct mat_alt {
    std::array<TypeNumber, col * row> data{}; // should be row major
    int32_t n_row{row};
    int32_t n_col{col};

    // most con & decon will use default
    mat_alt() = default;
    mat_alt(const mat_alt<TypeNumber, row, col>& other) = default;
    mat_alt(mat_alt<TypeNumber, row, col>&& other) noexcept = default;
    auto operator=(const mat_alt<TypeNumber, row, col>& other)
        -> mat_alt<TypeNumber, row, col>& = default;
    auto operator=(mat_alt<TypeNumber, row, col>&& other) noexcept
        -> mat_alt<TypeNumber, row, col>& = default;
    ~mat_alt() noexcept = default;

    // copy con & move con from std::array
    // check cppreference
    mat_alt(const std::array<TypeNumber, col * row>& other_data)
        : data(other_data) {}
    mat_alt(std::array<TypeNumber, col * row>&& other_data)
        : data(other_data) {}

    // copy con from mat (row major version)
    mat_alt(const mat<TypeNumber, row, col>& row_major)
        : n_row(row_major.n_row), n_col(row_major.n_col) {
        for (int32_t i = 0; i < row_major.n_row; ++i) {
            for (int32_t j = 0; j < row_major.n_col; ++j) {
                data[j * row_major.n_row + i] =
                    row_major.data[i * row_major.n_col + j];
            }
        }
    }
};

// bool operator ==, for testing
template <typename TypeNumber, int32_t row, int32_t col>
auto operator==(const mat<TypeNumber, row, col>& lv,
                const mat<TypeNumber, row, col>& rv) -> bool {
    return lv.data == rv.data;
}

template <typename TypeNumber, int32_t row_a, int32_t col_a, int32_t col_b>
auto matmul(const mat<TypeNumber, row_a, col_a>& A,
            const mat<TypeNumber, col_a, col_b>& B)
    -> std::unique_ptr<mat<TypeNumber, row_a, col_b>> {
    auto p_res = std::make_unique<mat<TypeNumber, row_a, col_b>>();
    auto res_n_row = p_res->n_row;
    auto res_n_col = p_res->n_col;

    // iterate over row and col
    for (int32_t i = 0; i < res_n_row; ++i) {
        for (int32_t j = 0; j < res_n_col; ++j) {
            TypeNumber acc{0};
            for (int32_t k = 0; k < A.n_col; ++k) {
                acc += A.data[i * A.n_col + k] *
                       B.data[k * B.n_col + j]; // A[i, k] * B[k, j]
            }
            p_res->data[i * res_n_col + j] = acc; // C[i, j]
        }
    }

    // should have moved? need to check cpprefernce
    return p_res;
}

template <typename TypeNumber, int32_t row_a, int32_t col_a, int32_t col_b>
auto matmul(const mat<TypeNumber, row_a, col_a>& A,
            const mat_alt<TypeNumber, col_a, col_b>& B)
    -> std::unique_ptr<mat<TypeNumber, row_a, col_b>> {
    auto p_res = std::make_unique<mat<TypeNumber, row_a, col_b>>();
    auto res_n_row = p_res->n_row;
    auto res_n_col = p_res->n_col;

    // iterate over row and col
    for (int32_t i = 0; i < res_n_row; ++i) {
        for (int32_t j = 0; j < res_n_col; ++j) {
            TypeNumber acc{0};
            for (int32_t k = 0; k < A.n_col; ++k) {
                acc += A.data[i * A.n_col + k] *
                       B.data[j * B.n_row + k]; // A[i, k] * B[k, j]
            }
            p_res->data[i * res_n_col + j] = acc; // C[i, j]
        }
    }

    // should have moved? need to check cpprefernce
    return p_res;
}

template <typename TypeNumber, int32_t row_a, int32_t col_a, int32_t col_b>
auto matmul_par(const mat<TypeNumber, row_a, col_a>& A,
                const mat<TypeNumber, col_a, col_b>& B, int32_t thread_count)
    -> std::unique_ptr<mat<TypeNumber, row_a, col_b>> {
    // unique_ptr thread safe problem, use raw pointer instead
    auto raw_p_res = new mat<TypeNumber, row_a, col_b>();
    auto res_n_row = raw_p_res->n_row;
    auto res_n_col = raw_p_res->n_col;

// iterate over row and col
// clang-format off
    #pragma omp parallel for num_threads(thread_count)
    // clang-format on
    for (int32_t i = 0; i < res_n_row; ++i) {
        for (int32_t j = 0; j < res_n_col; ++j) {
            TypeNumber acc{0};
            for (int32_t k = 0; k < A.n_col; ++k) {
                acc += A.data[i * A.n_col + k] *
                       B.data[k * B.n_col + j]; // A[i, k] * B[k, j]
            }
            raw_p_res->data[i * res_n_col + j] = acc; // C[i, j]
        }
    }

    // should have moved? need to check cpprefernce
    auto p_res = std::unique_ptr<mat<TypeNumber, row_a, col_b>>(raw_p_res);
    raw_p_res = nullptr; // manually transfer ownership
    return p_res;
}

template <typename TypeNumber, int32_t row_a, int32_t col_a, int32_t col_b>
auto matmul_par(const mat<TypeNumber, row_a, col_a>& A,
                const mat_alt<TypeNumber, col_a, col_b>& B,
                int32_t thread_count)
    -> std::unique_ptr<mat<TypeNumber, row_a, col_b>> {
    // unique_ptr thread safe problem, use raw pointer instead
    auto raw_p_res = new mat<TypeNumber, row_a, col_b>();
    auto res_n_row = raw_p_res->n_row;
    auto res_n_col = raw_p_res->n_col;

// iterate over row and col
// clang-format off
    #pragma omp parallel for num_threads(thread_count)
    // clang-format on
    for (int32_t i = 0; i < res_n_row; ++i) {
        for (int32_t j = 0; j < res_n_col; ++j) {
            TypeNumber acc{0};
            for (int32_t k = 0; k < A.n_col; ++k) {
                acc += A.data[i * A.n_col + k] *
                       B.data[j * B.n_row + k]; // A[i, k] * B[k, j]
            }
            raw_p_res->data[i * res_n_col + j] = acc; // C[i, j]
        }
    }

    // should have moved? need to check cpprefernce
    auto p_res = std::unique_ptr<mat<TypeNumber, row_a, col_b>>(raw_p_res);
    raw_p_res = nullptr; // manually transfer ownership
    return p_res;
}

} // namespace HW1P2
#endif