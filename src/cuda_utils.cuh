#pragma once

// use with care, see: https://stackoverflow.com/a/4433731
__forceinline__ __device__ unsigned lane_id() {
    unsigned ret;
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

#define FULL_MASK 0xffffffff
