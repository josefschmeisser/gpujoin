#pragma once

#include <array>
#include <cassert>
#include <utility>

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

template<class T1, class T2, class Compare = device_less<T1>>
__device__ device_size_t lower_bound(const T1& key, const T2* arr, const device_size_t size, Compare cmp = device_less<T1>{}) {
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

struct branchy_lower_bound_search_algorithm {
    //static constexpr char name[] = "branchy_lower_bound_search";
    static constexpr const char* name() {
        return "branchy_lower_bound_search";
    }

    template<class T1, class T2, class Compare = device_less<T1>>
    __device__ __forceinline__ device_size_t operator() (const T1& x, const T2* arr, const device_size_t size, Compare cmp = device_less<T1>{}) const {
        return lower_bound(x, arr, size, cmp);
    }
};

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

struct branchy_binary_search_algorithm {
    //static constexpr char name[] = "branchy_binary_search";
    static constexpr const char* name() {
        return "branchy_binary_search";
    }

    template<class T>
    __device__ __forceinline__ device_size_t operator() (T x, const T* arr, const device_size_t size) const {
        return branchy_binary_search(x, arr, size);
    }
};

template<class T>
struct device_clz {
};

template<>
struct device_clz<uint32_t> {
    __device__ __forceinline__ int operator() (const uint32_t x) const {
        return __clz(x);
    }
};

template<>
struct device_clz<uint64_t> {
    __device__ __forceinline__ int operator() (const uint64_t x) const {
        return __clzll(x);
    }
};

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
    static constexpr unsigned bit_count = sizeof(device_size_t)*8u - 1u;
    const unsigned steps = bit_count - device_clz<device_size_t>{}(size - 1);
    device_size_t mid = static_cast<device_size_t>(1u) << steps;
    device_size_t ret = (arr[mid] < x) * (size - mid);
    for (unsigned step = 1; step <= steps; ++step) {
        mid >>= 1;
        assert(ret + mid < size);
        ret += (arr[ret + mid] < x) ? mid : 0;
    }
    ret += (arr[ret] < x) ? 1 : 0;

    return ret;
}

struct branch_free_binary_search_algorithm {
    //static constexpr char name[] = "branch_free_binary_search";
    static constexpr const char* name() {
        return "branch_free_binary_search";
    }

    template<class T>
    __device__ __forceinline__ device_size_t operator() (T x, const T* arr, const device_size_t size) const {
        return branch_free_binary_search(x, arr, size);
    }
};

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

/*
template<unsigned... Is>
__device__ constexpr auto split_impl(std::integer_sequence<unsigned, Is...> s, unsigned window_size) {// -> std::array<unsigned, s.size()> {
    std::array<unsigned, sizeof...(Is)> offsets{};
    //((offsets[Is] = Is*window_size/s.size()), ...);
    //(void) (int[]) {(offsets[Is] = Is*window_size/s.size(), 0)...};
    (void) (int[]) {(offsets[Is] = Is*window_size/sizeof...(Is), 0)...};
    return offsets;
}

template<unsigned Co_Op_Extent, unsigned WindowSize>
__device__ constexpr std::array<unsigned, Co_Op_Extent> split() {
    return split_impl(std::make_integer_sequence<unsigned, Co_Op_Extent>{}, WindowSize);
}

template<class T, size_t size>
__device__ const T& array_at(const std::array<T, size>& a, size_t idx) {
    return *(reinterpret_cast<const T*>(&a));
}
*/

template<unsigned Co_Op_Extent, unsigned WindowLength>
struct splitter {
    __device__ constexpr splitter() : arr() {
        const unsigned step = WindowLength/Co_Op_Extent;
        for (unsigned i = 0; i < Co_Op_Extent - 1; ++i) {
            arr[i] = i*step;
        }
        arr[Co_Op_Extent - 1] = WindowLength - 1;
        arr[Co_Op_Extent] = WindowLength; // sentinel element, used to simplify the search logic
    }

    __device__ constexpr unsigned operator[] (unsigned pos) const {
        return arr[pos];
    }

    unsigned arr[Co_Op_Extent + 1];
};

template<class T, unsigned Co_Op_Extent, unsigned WindowSize>
__device__ __forceinline__ device_size_t cooperative_binary_search_stride(bool is_leader, T x, const T* arr, const device_size_t size, const uint32_t group_mask) {
//printf("input array %p\n", arr);
//assert(arr & (__ffs(sizeof(T)) - 1) == 0);
    const unsigned my_lane_id = lane_id();
    const unsigned thread_offset = my_lane_id - (__ffs(group_mask) - 1);
    //constexpr auto window_offsets = split<Co_Op_Extent, WindowSize/sizeof(T)>();
    //static const std::array<unsigned, Co_Op_Extent> window_offsets{}; // split<Co_Op_Extent, WindowSize/sizeof(T)>();
    constexpr unsigned window_length = WindowSize/sizeof(T);
    constexpr auto window_offsets = splitter<Co_Op_Extent, WindowSize/sizeof(T)>();
    const unsigned thread_window_offset = window_offsets[thread_offset]; // array_at(window_offsets, thread_offset);// window_offsets[thread_offset];
//printf("lane: %u thread_window_offset: %u\n", my_lane_id, thread_window_offset);

    uint32_t matches_mask = 0u;
    device_size_t lower = 0;
    device_size_t count = size;

    while (count > 0) {
        const device_size_t step = count / 2;
        const device_size_t mid = (lower + step) & ~ static_cast<device_size_t>(window_length - 1); // round down to next multiple of window_length

        /*
max_stream_portion = (max_stream_portion + ALIGN_BYTES - 1) & -ALIGN_BYTES;
        */
//const auto aligned_mid = &(arr[mid] & (WindowSize - 1))
        //const device_size_t pos = min(window_offset + mid - (mid & WindowSize), size - 1); // align to cache line boundary // TODO check
       // const device_size_t pos = min(window_offset + mid, size - 1); // align to cache line boundary // TODO check

//printf("lane: %u: step: %lu, original_mid: %lu, mid: %lu\n", my_lane_id, step, lower + step, mid);
        const auto lane_mid = min(mid + thread_window_offset, size - 1);
        const auto r = arr[lane_mid] < x; // TODO cmp(arr[pos], x);
        matches_mask = __ballot_sync(group_mask, r);
//if (thread_offset == 0) printf("lane: %u: group_mask: %x, matches_mask: %x\n", my_lane_id, group_mask, matches_mask);
        if (matches_mask == group_mask) {
            lower = mid + window_length;
            count -= step + 1;
        } else if (matches_mask == 0u) {
            count = step;
        } else {
            // use a branch free binary search from here on
            lower = mid;
            break;
        }
//break;
    }
/*
//    count = window_offsets[__clz(matches_mask) - __clz(group_mask)];
//const unsigned first_matching_lane = __clz(matches_mask) - __clz(group_mask);
//const unsigned first_matching_lane = __ffs(group_mask) - __ffs(matches_mask);
const unsigned first_matching_lane = (Co_Op_Extent - 1) - (__clz(matches_mask) - __clz(group_mask));
count = window_offsets[first_matching_lane + 1]; // advance to the first missmatch; use sentinel if necessary
//printf("lane: %u: first_matching_lane: %u, count: %u, lower: %lu\n", my_lane_id, first_matching_lane, count, lower);
    // TODO count = array_at(window_offsets, __clz(matches_mask) - __clz(group_mask));

    if (is_leader) {
assert(lower + count <= size);
        //lower = lower + branch_free_binary_search(x, arr + lower, size - lower);
        lower = lower + branch_free_binary_search(x, arr + lower, count);
//printf("lane: %u: x: %lu, lower: %lu, arr[lower]: %lu\n", my_lane_id, x, lower, arr[lower]);
        assert(arr[lower] == x);
    }
*/

    if (is_leader) {
const unsigned first_matching_lane = (Co_Op_Extent - 1) - (__clz(matches_mask) - __clz(group_mask));
lower += window_offsets[first_matching_lane];
device_size_t upper = window_offsets[first_matching_lane + 1]; // advance to the first missmatch; use sentinel if necessary
upper = min(size - lower, upper);
//printf("lane: %u: x: %lu, arr[lower]: %lu, arr[upper]: %lu\n", my_lane_id, x, arr[lower + bound_l], arr[lower + upper]);
lower = lower + linear_search(x, arr + lower, upper);
        assert(arr[lower] == x);
    }
    
    return lower;

    // alternatively: linear search with all threads
}


template<class T, unsigned Co_Op_Degree = 3, unsigned WindowSize = GPU_CACHE_LINE_SIZE>
__device__ device_size_t cooperative_binary_search(bool active, T x, const T* arr, const device_size_t size) {
    enum { THREAD_GROUP_SIZE  = 1 << Co_Op_Degree };

    const unsigned my_lane_id = lane_id();
    unsigned first_thread = THREAD_GROUP_SIZE*(my_lane_id >> Co_Op_Degree);
    device_size_t lower_bound = size;
    const uint32_t group_mask = __funnelshift_l(FULL_MASK, 0, THREAD_GROUP_SIZE) << first_thread;

    //static constexpr uint32_t group_mask = ((1u << THREAD_GROUP_SIZE) - 1u) << leader;

    assert(my_lane_id >= first_thread);
    const int lane_offset = my_lane_id - first_thread;

//printf("lane: %u x: %lu\n", my_lane_id, x);
    for (unsigned shift = 0; shift < THREAD_GROUP_SIZE; ++shift) {
        // advance leader position within thread group
        const unsigned leader = first_thread + shift;
        const auto leader_active = __shfl_sync(group_mask, active, leader);
        if (!leader_active) continue;

//        int key_idx = lane_offset - THREAD_GROUP_SIZE;
        const T leader_x = __shfl_sync(group_mask, x, leader);
        const T* leader_arr = reinterpret_cast<const T*>(__shfl_sync(group_mask, reinterpret_cast<uint64_t>(arr), leader));
        const device_size_t leader_size = __shfl_sync(group_mask, size, leader);

        const bool is_leader = my_lane_id == leader;
        const auto leader_lower_bound = cooperative_binary_search_stride<T, THREAD_GROUP_SIZE, WindowSize>(is_leader, leader_x, leader_arr, leader_size, group_mask);
        if (is_leader) {
            lower_bound = leader_lower_bound;
        }

        //break;
    }
assert(arr[lower_bound] == x);
    assert(!active || lower_bound <= size || arr[lower_bound] >= x);
    return lower_bound;
}

struct cooperative_binary_search_algorithm {
    static constexpr const char* name() {
        return "cooperative_binary_search";
    }

    template<class T>
    __device__ __forceinline__ device_size_t operator() (bool active, T x, const T* arr, const device_size_t size) const {
        return cooperative_binary_search(active, x, arr, size);
    }
};
