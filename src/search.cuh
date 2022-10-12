#pragma once

#include <cassert>

#include "device_definitions.hpp"
#include "cuda_utils.cuh"

template<class T>
struct device_less {
    // Declaration of the less operation
    __device__ __forceinline__ bool operator() (const T& x, const T& y) const {
        return x < y;
    }
};

template<class T>
struct device_greater {
    // Declaration of the less operation
    __device__ __forceinline__ bool operator() (const T& x, const T& y) const {
        return x > y;
    }
};

template<class T, class Compare = less<T>>
__device__ device_size_t lower_bound(const T& key, const T* arr, const device_size_t size, Compare cmp = device_less<T>{}) {
    device_size_t lower = 0;
    device_size_t count = size;
    while (count > 0) {
        device_size_t step = count / 2;
        device_size_t mid = lower + step;
        if (cmp(arr[mid], key)) {
            lower = mid + 1;
            count -= step + 1;
        } else {
            count = step;
        }
    }
    return lower;
}
/*
template<class T, class Compare = less<T>>
__device__ device_size_t branchy_binary_search(T x, const T* arr, const device_size_t size, Compare cmp = less<T>{}) {
//__device__ device_size_t branchy_binary_search(T x, const T* arr, const device_size_t size, Compare cmp) {
    device_size_t lower = 0;
    device_size_t upper = size;
    do {
        device_size_t mid = ((upper - lower) / 2) + lower;
        const int c = cmp(x, arr[mid]);
        if (c < 0) {
            upper = mid;
        } else if (c > 0) {
            lower = mid + 1;
        } else {
            return mid;
        }
    } while (lower < upper);
    return lower;
}
*/

/**
 * Binary Search.
 *
 * A branch-based text book binary search implementation.
 *
 * @param x Value to search for.
 * @param arr Search array..
 * @param size Size of `arr`.
 * @return Position of `x` in `arr`.
 */
template<class T>
__device__ device_size_t branchy_binary_search(T x, const T* arr, const device_size_t size) {
    device_size_t lower = 0;
    device_size_t upper = size;
    do {
        device_size_t mid = ((upper - lower) / 2) + lower;
        if (x < arr[mid]) {
            upper = mid;
        } else if (x > arr[mid]) {
            lower = mid + 1;
        } else {
            return mid;
        }
    } while (lower < upper);
    return lower;
}

/**
 * Binary Search.
 *
 * A branch-free binary search implementation.
 *
 * @param x Value to search for.
 * @param arr Search array..
 * @param size Size of `arr`.
 * @return lower bound position of `x` in `arr`.
 */
template<class T>
__device__ device_size_t branch_free_binary_search(T x, const T* arr, const device_size_t size) { // TODO rename? should be lower bound?
    if (size < 1) { return 0; }

    const unsigned steps = 31u - __clz(size - 1);
    device_size_t mid = 1 << steps;
    device_size_t ret = (arr[mid] < x) * (size - mid);
    for (unsigned step = 1; step <= steps; ++step) {
        mid >>= 1;
        ret += (arr[ret + mid] < x) ? mid : 0;
    }
    ret += (arr[ret] < x) ? 1 : 0;

    return ret;
}

template<class T, unsigned max_steps = 4> // TODO find optimal limit
__device__ device_size_t branch_free_exponential_search(T x, const T* arr, const device_size_t n, const float hint) {
    //if (size < 1) return;

    const int last = n - 1;
    const int start = static_cast<int>(last*hint);
    assert(start <= last);

    bool cont = true;
    bool less = arr[start] < x;
    int offset = -1 + 2*less;
    device_size_t current = max(0, min(last , start + offset));
    for (unsigned i = 0; i < max_steps; ++i) {
        cont = ((arr[current] < x) == less);
        offset = cont ? offset<<1 : offset;
        current = max(0, min(last, start + offset));
    }

    const auto pre_lower = max(0, min(static_cast<int>(n), start + (offset>>less)));
    const auto pre_upper = 1 + max(0, min(static_cast<int>(n), start + (offset>>(1 - less))));
    const device_size_t lower = (!cont || less) ? pre_lower : 0;
    const device_size_t upper = (!cont || !less) ? pre_upper : n;

    return lower + branchy_binary_search(x, arr + lower, upper - lower); // TODO measure alternatives
    //return lower + branch_free_binary_search(x, arr + lower, upper - lower); // TODO measure alternatives
}

template<class T>
__device__ device_size_t exponential_search(T x, const T* arr, const device_size_t size) {
    assert(size > 0);
    device_size_t bound = 1;
    while (bound < size && arr[bound] < x) {
        bound <<= 1;
    }
    const device_size_t lower = bound>>1;
    return lower + branchy_binary_search(x, arr + lower, min(bound + 1, size - lower));
}

template<class T>
__device__ device_size_t linear_search(T x, const T* arr, const device_size_t size) {
    for (device_size_t i = 0; i < size; ++i) {
        if (arr[i] >= x) return i;
    }
    return size;
}

template<class T, unsigned Co_Op_Degree = 3>
__device__ device_size_t cooperative_linear_search(bool active, T x, const T* arr, const device_size_t size) {
    enum { WINDOW_SIZE = 1 << Co_Op_Degree };

    const unsigned my_lane_id = lane_id();
    unsigned leader = WINDOW_SIZE*(my_lane_id >> Co_Op_Degree);
    device_size_t lower_bound = size;
    const uint32_t window_mask = __funnelshift_l(FULL_MASK, 0, WINDOW_SIZE) << leader; // TODO replace __funnelshift_l() with compile time computation
    assert(my_lane_id >= leader);
    const int lane_offset = my_lane_id - leader;

    for (unsigned shift = 0; shift < WINDOW_SIZE; ++shift) {
        int key_idx = lane_offset - WINDOW_SIZE;
        const T leader_x = __shfl_sync(window_mask, x, leader);
        const T* leader_arr = reinterpret_cast<const T*>(__shfl_sync(window_mask, reinterpret_cast<uint64_t>(arr), leader));
        const device_size_t leader_size = __shfl_sync(window_mask, size, leader);

        const auto leader_active = __shfl_sync(window_mask, active, leader);
        bool advance = leader_active;
        uint32_t matches = 0;
        while (matches == 0 && advance) {
            key_idx += WINDOW_SIZE;

            T value;
            if (key_idx < leader_size) value = leader_arr[key_idx];
            matches = __ballot_sync(window_mask, key_idx < leader_size && value >= leader_x);
            advance = key_idx - lane_offset + WINDOW_SIZE < size; // termination criterion
        }

        if (my_lane_id == leader && matches != 0) {
            lower_bound = key_idx + __ffs(matches) - 1 - leader;
        }

        leader += 1;
    }

    assert(!active || lower_bound >= size || arr[lower_bound] >= x);
    return lower_bound;
}

template<unsigned... Is>
__device__ constexpr auto split_impl(std::integer_sequence<unsigned, Is...> s, unsigned window_size) -> std::array<unsigned, s.size()> {
    std::array<int, s.size()> offsets{};
    ((offsets[Is] = Is*window_size/s.size()), ...);
    return offsets;
}

template<unsigned Co_Op_Extent, unsigned WindowSize = CPU_CACHE_LINE_SIZE>
__device__ constexpr std::array<unsigned, Co_Op_Extent> split() {
    return split_impl(std::make_integer_sequence<unsigned, Co_Op_Extent>{}, WINDOW_SIZE);
}

template<class T, unsigned Co_Op_Extent, unsigned WindowSize = CPU_CACHE_LINE_SIZE>
__device__ __forceinline__ device_size_t cooperative_binary_search_stride(T x, const T* arr, const device_size_t size, const uint32_t group_mask) {
    const unsigned my_lane_id = lane_id();
    const unsigned thread_offset = my_lande_id - __ffs(group_mask);
    static constexpr auto window_offsets = split();
    static constexpr auto window_offset = window_offsets[thread_offset];

    uint32_t matches_mask = 0u;
    device_size_t lower = 0;
    device_size_t count = size;
    while (count > 0) {
        const device_size_t step = count / 2;
        const device_size_t mid = lower + step;
        const device_size_t pos = min(window_offset + mid - (mid & WindowSize), size - 1); // align to cache line boundary // TODO check
        const auto r = cmp(arr[pos], key);
        matches_mask = __ballot_sync(group_mask, r);

        if (matches == group_mask) {
            lower = mid + 1;
            count -= step + 1;
        } else if (matches == 0u) {
            count = step;
        } else {
            // use a branch free binary search from here on
            break;
        }
    }

    count = thread_offsets[__clz(matches_mask) - __clz(group_mask)];
    return branch_free_binary_search(x, arr + lower, count);

    // alternatively: linear search with all threads
}

template<class T, unsigned Co_Op_Degree = 3, unsigned WindowSize = CPU_CACHE_LINE_SIZE>
__device__ device_size_t cooperative_binary_search(bool active, T x, const T* arr, const device_size_t size) {
    enum { THREAD_GROUP_SIZE  = 1 << Co_Op_Degree };

    const unsigned my_lane_id = lane_id();
    unsigned leader = THREAD_GROUP_SIZE*(my_lane_id >> degree);
    device_size_t lower_bound = size;
    const uint32_t group_mask = __funnelshift_l(FULL_MASK, 0, THREAD_GROUP_SIZE) << leader;

    //static constexpr uint32_t group_mask = ((1u << THREAD_GROUP_SIZE) - 1u) << leader;

    assert(my_lane_id >= leader);
    const int lane_offset = my_lane_id - leader;

    for (unsigned shift = 0; shift < THREAD_GROUP_SIZE; ++shift) {
        int key_idx = lane_offset - THREAD_GROUP_SIZE;
        const T leader_x = __shfl_sync(group_mask, x, leader);
        const T* leader_arr = reinterpret_cast<const T*>(__shfl_sync(group_mask, reinterpret_cast<uint64_t>(arr), leader));
        const device_size_t leader_size = __shfl_sync(group_mask, size, leader);

        const auto leader_active = __shfl_sync(group_mask, active, leader);
        if (leader_active) {
            const auto leader_lower_bound = cooperative_binary_search_stride<T, THREAD_GROUP_SIZE, WindowSize>(leader_x, leader_arr, leader_size, group_mask);
            if (my_lane_id == leader) {
                lower_bound = leader_lower_bound;
            }
        }

        // advance leader position within thread group
        leader += 1;
    }

    assert(!active || lower_bound >= size || arr[lower_bound] >= x);
    return lower_bound;
}
