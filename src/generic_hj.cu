#include "index_lookup_config.hpp"
#include "generic_hj.cuh"

#include "cuda_utils.cuh"

template<class KeyType>
__global__ void hj_build_kernel(const hj_args args) {
    //printf("hj_build_kernel\n");
    const KeyType* __restrict__ build_side_rel = reinterpret_cast<const KeyType*>(args.build_side_rel);

    auto& ht = args.state->ht;

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < args.build_side_size; i += stride) {
        //printf("build_side_rel[i]: %lu i: %lu\n", build_side_rel[i], i);
        ht.insert(build_side_rel[i], i);
    }
}

template<class KeyType>
__global__ void hj_probe_kernel(const hj_args args) {
    const KeyType* __restrict__ probe_side_rel = reinterpret_cast<const KeyType*>(args.probe_side_rel);
    const KeyType* __restrict__ build_side_rel = reinterpret_cast<const KeyType*>(args.build_side_rel);
    value_t* __restrict__ tids = args.tids;

    auto& ht = args.state->ht;

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    //printf("k: %lu tid: %lu\n", args.probe_side_size);
    for (int i = index; i < args.probe_side_size; i += stride) {
        device_size_t build_side_tid;
        const auto k = probe_side_rel[i];
        bool match = ht.lookup(k, build_side_tid);
        if (match) {
            //printf("k: %lu build_side_tid: %lu\n", k, build_side_tid);
            //tids[build_side_tid] = build_side_rel[build_side_tid];
            tids[build_side_tid] = k;
        }
    }
}

template __global__ void hj_build_kernel<index_key_t>(const hj_args args);
template __global__ void hj_probe_kernel<index_key_t>(const hj_args args);
