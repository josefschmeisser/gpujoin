#include <sys/types.h>

#include "cuda_utils.cuh"
#include "generic_hj.cuh"
#include "index_lookup_config.hpp"

template<class KeyType>
__global__ void hj_build_kernel(const hj_args args) {
    const KeyType* __restrict__ build_side_rel = reinterpret_cast<const KeyType*>(args.build_side_rel);

    auto& ht = args.state->ht;

    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < args.build_side_size; i += stride) {
        ht.insert(build_side_rel[i], i);
    }
}

template<class KeyType>
__global__ void hj_probe_kernel(const hj_args args) {
    const KeyType* __restrict__ probe_side_rel = reinterpret_cast<const KeyType*>(args.probe_side_rel);
    value_t* __restrict__ tids = args.tids;

    auto& ht = args.state->ht;

#ifdef ONLY_AGGREGATES
    KeyType thread_acc;
    __shared__ KeyType block_acc;
#endif

    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = blockDim.x * gridDim.x;
    for (device_size_t i = index; i < args.probe_side_size; i += stride) {
        device_size_t build_side_tid;
        const auto k = probe_side_rel[i];
        bool match = ht.lookup(k, build_side_tid);
        if (match) {
#ifdef ONLY_AGGREGATES
            thread_acc += k;
#else
            tids[build_side_tid] = k;
#endif
        }
    }

#ifdef ONLY_AGGREGATES
    // reduce accumulator
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_acc += __shfl_down_sync(FULL_MASK, thread_acc, offset);
    }
    if (lane_id() == 0) {
        tmpl_atomic_add(&block_acc, thread_acc);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        int offset = (blockIdx.x * blockDim.x) % args.build_side_size;
        tmpl_atomic_add(tids + offset, block_acc);
    }
#endif
}

template __global__ void hj_build_kernel<index_key_t>(const hj_args args);
template __global__ void hj_probe_kernel<index_key_t>(const hj_args args);
