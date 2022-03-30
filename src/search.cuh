#pragma once

#include <cassert>

#include "cuda_utils.cuh"

template<class T>
__device__ unsigned branchy_binary_search(T x, const T* arr, const unsigned size) {
    unsigned lower = 0;
    unsigned upper = size;
    do {
        unsigned mid = ((upper - lower) / 2) + lower;
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

template<class T>
__device__ unsigned branch_free_binary_search(T x, const T* arr, const unsigned size) {
    if (size < 1) { return 0; }

    const unsigned steps = 31 - __clz(size - 1);
    //printf("steps: %d\n", steps);
    unsigned mid = 1 << steps;

    unsigned ret = (arr[mid] < x) * (size - mid);
    //while (mid > 0) {
    for (unsigned step = 1; step <= steps; ++step) {
        mid >>= 1;
        ret += (arr[ret + mid] < x) ? mid : 0;
    }
    ret += (arr[ret] < x) ? 1 : 0;

    return ret;
}

template<class T, unsigned max_step = 4> // TODO find optimal limit
__device__ unsigned branch_free_exponential_search(T x, const T* arr, const unsigned n, const float hint) {
    //if (size < 1) return;

    const int last = n - 1;
    const int start = static_cast<int>(last*hint);
    assert(start <= last);

    bool cont = true;
    bool less = arr[start] < x;
    int offset = -1 + 2*less;
    unsigned current = max(0, min(last , start + offset));
    for (unsigned i = 0; i < max_step; ++i) {
        cont = ((arr[current] < x) == less);
        offset = cont ? offset<<1 : offset;
        current = max(0, min(last, start + offset));
    }

    const auto pre_lower = max(0, min(static_cast<int>(n), start + (offset>>less)));
    const auto pre_upper = 1 + max(0, min(static_cast<int>(n), start + (offset>>(1 - less))));
    const unsigned lower = (!cont || less) ? pre_lower : 0;
    const unsigned upper = (!cont || !less) ? pre_upper : n;

    return lower + branchy_binary_search(x, arr + lower, upper - lower); // TODO measure alternatives
//    return lower + branch_free_binary_search(x, arr + lower, upper - lower); // TODO measure alternatives
}

template<class T>
__device__ unsigned exponential_search(T x, const T* arr, const unsigned size) {
    assert(size > 0);
    int bound = 1;
    while (bound < size && arr[bound] < x) {
        bound <<= 1;
    }
    const auto lower = bound>>1;
    return lower + branchy_binary_search(x, arr + lower, min(bound + 1, size - lower));
}

template<class T>
__device__ unsigned linear_search(T x, const T* arr, const unsigned size) {
    for (unsigned i = 0; i < size; ++i) {
        if (arr[i] >= x) return i;
    }
    return size;
}

template<class T, unsigned degree = 3>
__device__ unsigned cooperative_linear_search(bool active, T x, const T* arr, const unsigned size) {

    enum { WINDOW_SIZE = 1 << degree };

//if (threadIdx.x < 832 || threadIdx.x > 863) return 0;
    assert(__all_sync(FULL_MASK, 1));


    const unsigned my_lane_id = lane_id();

    unsigned lower_bound = size;

    //uint32_t leader = 1 << degree*(my_lane_id >> degree);
    unsigned leader = WINDOW_SIZE*(my_lane_id >> degree);
  //  printf("thread: %d lane: %d leader: %d\n", threadIdx.x, my_lane_id, leader);
    //__funnelshift_l ( unsigned int  lo, unsigned int  hi, unsigned int  shift )
    const uint32_t window_mask = __funnelshift_l(FULL_MASK, 0, WINDOW_SIZE) << leader; // TODO replace __funnelshift_l() with compile time computation
  //  printf("thread: %d lane: %d window_mask: 0x%.8X\n", threadIdx.x, my_lane_id, window_mask);

    assert(my_lane_id >= leader);
    const int lane_offset = my_lane_id - leader;
    //const int lane_offset = max(my_lane_id, leader) - min(my_lane_id, leader); // TODO

    for (unsigned shift = 0; shift < WINDOW_SIZE; ++shift) {
        int key_idx = lane_offset - WINDOW_SIZE;
        const T leader_x = __shfl_sync(window_mask, x, leader);
        const T* leader_arr = reinterpret_cast<const T*>(__shfl_sync(window_mask, reinterpret_cast<uint64_t>(arr), leader));
        const unsigned leader_size = __shfl_sync(window_mask, size, leader);

        const auto leader_active = __shfl_sync(window_mask, active, leader);
        unsigned exhausted_cnt = leader_active ? 0 : WINDOW_SIZE;

  //      if (my_lane_id == leader && !leader_active) printf("thread: %d leader: %d leader not active\n", threadIdx.x, leader);

        uint32_t matches = 0;
        while (matches == 0 && exhausted_cnt < WINDOW_SIZE) {
            key_idx += WINDOW_SIZE;

            T value;
            if (key_idx < leader_size) value = leader_arr[key_idx];
            matches = __ballot_sync(window_mask, key_idx < leader_size && value >= leader_x);
            exhausted_cnt = __popc(__ballot_sync(window_mask, key_idx >= leader_size));

   //         if (leader == 8) printf("thread: %d leader: %d key_idx: %d value: %d\n", threadIdx.x, leader, key_idx, value);
  //          if (my_lane_id == leader) printf("thread: %d leader: %d matches: 0x%.8X exhausted_cnt: %d\n", threadIdx.x, leader, matches, exhausted_cnt);
        }

        if (my_lane_id == leader && matches != 0) {
    //        printf("thread: %d lane: %d key_idx: %u, ffs: %u\n", threadIdx.x, my_lane_id, key_idx, __ffs(matches) - 1 - leader);
            lower_bound = key_idx + __ffs(matches) - 1 - leader;
        } else if (my_lane_id == leader) {
  //          printf("thread: %d lane: %d key_idx: %u\n", threadIdx.x, my_lane_id, lower_bound);
        }

        leader += 1;
    }
  //  printf("thread: %d lane: %d size: %u lower_bound: %u arr[lower_bound]: %u x: %u arr[size - 1]: %u\n", threadIdx.x, my_lane_id, size, lower_bound, arr[lower_bound], x, arr[size - 1]);
    assert(!active || lower_bound >= size || arr[lower_bound] >= x);
    return lower_bound;
}
