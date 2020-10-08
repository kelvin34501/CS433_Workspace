#ifndef HW1P1_H
#define HW1P1_H

#include <iterator>

template <typename Iter> auto sum_iter(Iter begin, Iter end) -> decltype(auto) {
    typename std::iterator_traits<Iter>::value_type res{};

    for (auto it = begin; it != end; ++it) {
        res = res + *it;
    }

    return res;
}

template <typename Iter>
auto sum_iter_par(Iter begin, Iter end, int32_t thread_count)
    -> decltype(auto) {
    typename std::iterator_traits<Iter>::value_type res{};

    // clang-format off
    #pragma omp parallel for num_threads(thread_count)
    // clang-format on
    for (auto it = begin; it != end; ++it) {
        // clang-format off
        #pragma omp critical
        // clang-format on
        res = res + *it;
    }

    return res;
}

template <typename Iter>
auto sum_iter_par_red(Iter begin, Iter end, int32_t thread_count)
    -> decltype(auto) {
    typename std::iterator_traits<Iter>::value_type res{};

    // clang-format off
    #pragma omp parallel for num_threads(thread_count) reduction(+ : res)
    // clang-format on
    for (auto it = begin; it != end; ++it) {
        res = res + *it;
    }

    return res;
}

#endif