#ifndef HW1P1_H
#define HW1P1_H

#include <iterator>
#include <omp.h>

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
    typename std::iterator_traits<Iter>::difference_type diff = end - begin;

    // clang-format off
    #pragma omp parallel for num_threads(thread_count) shared(diff, begin, res) default(none)
    // clang-format on
    for (auto it = 0; it < diff; ++it) {
        // clang-format off
        #pragma omp critical
        // clang-format on
        res = res + *(begin + it);
    }

    return res;
}

template <typename Iter>
auto sum_iter_par_lock(Iter begin, Iter end, int32_t thread_count)
    -> decltype(auto) {
    typename std::iterator_traits<Iter>::value_type res{};
    typename std::iterator_traits<Iter>::difference_type diff = end - begin;

    omp_lock_t writelock;

    omp_init_lock(&writelock);

    // clang-format off
    #pragma omp parallel for num_threads(thread_count) shared(diff, begin, res, writelock) default(none)
    // clang-format on
    for (auto it = 0; it < diff; ++it) {
        omp_set_lock(&writelock);
        res = res + *(begin + it);
        omp_unset_lock(&writelock);
    }

    return res;
}

template <typename Iter>
auto sum_iter_par_red(Iter begin, Iter end, int32_t thread_count)
    -> decltype(auto) {
    typename std::iterator_traits<Iter>::value_type res{};
    typename std::iterator_traits<Iter>::difference_type diff = end - begin;

    // clang-format off
    #pragma omp parallel for num_threads(thread_count) reduction(+ : res) shared(diff, begin) default(none)
    // clang-format on
    for (auto it = 0; it < diff; ++it) {
        res = res + *(begin + it);
    }

    return res;
}

#endif