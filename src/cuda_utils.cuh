#pragma once

// use with care, see: https://stackoverflow.com/a/4433731
__forceinline__ __device__ unsigned lane_id() {
    unsigned ret;
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

template<class T>
__device__ T atomic_add_sat(T* address, T val, T saturation) {
    unsigned expected, update, old;
    old = *address;
    do {
        expected = old;
        update = (old + val > saturation) ? saturation : old + val;
        old = atomicCAS(address, expected, update);
    } while (expected != old);
    return old;
}

template<class T>
__device__ T atomic_sub_safe(T* address, T val) {
    unsigned expected, update, old;
    old = *address;
    do {
        expected = old;
        update = (old > val) ? (old - val) : 0;
        old = atomicCAS(address, expected, update);
    } while (expected != old);
    return old;
}

template<class T>
__forceinline__ __device__ T round_up_pow2(T value) {
    return static_cast<T>(1) << (sizeof(T)*8 - __clz(value - 1));
}

#define FULL_MASK 0xffffffff
#define GPU_CACHE_LINE_SIZE 128
