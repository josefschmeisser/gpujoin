#include "index_lookup_config.hpp"
#include "generic_hj.cuh"

#include "cuda_utils.cuh"

template<class KeyType>
__global__ void hj_build_kernel(const hj_args args) {
    const KeyType* __restrict__ build_side_rel = reinterpret_cast<const KeyType*>(args.build_side_rel);

    auto& ht = args.state->ht;

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < args.build_side_size; i += stride) {
        ht.insert(build_side_rel[i], i);
    }
}

template<class KeyType>
__global__ void hj_probe_kernel(const hj_args args) {
    const KeyType* __restrict__ probe_side_rel = reinterpret_cast<const KeyType*>(args.probe_side_rel);
    auto& ht = args.state->ht;

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = index; i < args.probe_side_size; i += stride) {
        device_size_t tid;
        bool match = ht.lookup(probe_side_rel[i], tid);
        // TODO use lane refill
        if (match) {
            args.tids[index] = tid;
        }
    }
}

template __global__ void hj_build_kernel<index_key_t>(const hj_args args);
template __global__ void hj_probe_kernel<index_key_t>(const hj_args args);
