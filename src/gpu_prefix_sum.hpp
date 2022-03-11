#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace gpu_prefix_sum {

// same as in partition.rs
uint32_t fanout(uint32_t radix_bits) {
    return (1 << radix_bits);
}

template<class G, class B>
size_t state_size(G grid_size, B block_size) {
    cudaDeviceProp device_properties;
    const auto ret = cudaGetDeviceProperties(&device_properties, 0); // FIXME
    CubDebugExit(ret);

    const auto warp_size = device_properties.warpSize;
    return ((grid_size * block_size) / warp_size + warp_size);
}

} // end namespace gpu_prefix_sum
