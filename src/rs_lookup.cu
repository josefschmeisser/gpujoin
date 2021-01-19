#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <chrono>
#include <cstring>
#include <stdexcept>

#include "rs/multi_map.h"
#include "btree.cuh"
#include "zipf.hpp"
#include "utils.hpp"

#include "rs.cu"

using namespace std;
using namespace rs;

using payload_t = btree::payload_t;
using rs_placement_policy = vector_to_device_array;// vector_to_managed_array;

static constexpr int device_id = 0;
static constexpr unsigned maxRepetitions = 10;
static constexpr unsigned numElements = 1e8;
static constexpr payload_t invalidTid = std::numeric_limits<payload_t>::max();

struct Relation {
    size_t count;
    rs_key_t* pk;
    uint64_t* payload;
};

__device__ payload_t rs_lookup(const DeviceRadixSpline* __restrict__ rs, const rs_key_t key, const Relation& rel) {
    const unsigned estimate = rs::cuda::get_estimate(rs, key);
//    printf("key: %lu estimate: %u\n", key, estimate);
    const unsigned begin = (estimate < rs->max_error_) ? 0 : (estimate - rs->max_error_);
    const unsigned end = (estimate + rs->max_error_ + 2 > rs->num_keys_) ? rs->num_keys_ : (estimate + rs->max_error_ + 2);

    const auto bound_size = end - begin;
    const unsigned pos = begin + rs::cuda::lower_bound(key, &rel.pk[begin], bound_size, [] (const rs_key_t& a, const rs_key_t& b) -> int {
        return a < b;
    });
//    int index = blockIdx.x * blockDim.x + threadIdx.x;
//    printf("key: %lu search bound [%u, %u) pos: %u expected: %d\n", key, begin, end, pos, index);
    return (pos < rel.count) ? static_cast<payload_t>(pos) : invalidTid;
}

__global__ void rs_bulk_lookup(const DeviceRadixSpline* __restrict__ rs, unsigned n, const rs_key_t* __restrict__ keys, Relation rel, payload_t* __restrict__ tids) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        tids[i] = rs_lookup(rs, keys[i], rel);
    }
}

__managed__ bool valid = true;
__global__ void validate_results(const payload_t* __restrict__ tids, unsigned n, const rs_key_t* __restrict__ keys, Relation rel) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        auto pos = reinterpret_cast<size_t>(tids[i]);
        if (keys[i] != rel.pk[pos]) {
            printf("mismatch\n");
            valid = false;
        }
    }
}

int main(int argc, char** argv) {
    (void)device_id;

    Relation rel;
    rs_key_t* lookupKeys;
    DeviceRadixSpline* d_rs;

    {
        // Create random keys.
        vector<rs_key_t> keys(numElements);
        generate(keys.begin(), keys.end(), rand);
        sort(keys.begin(), keys.end());

        auto rs = build_radix_spline(keys);
        d_rs = rs::copy_radix_spline<rs_placement_policy>(rs);

        const auto keys_size = sizeof(rs_key_t)*keys.size();
        cudaMallocManaged(&rel.pk, keys_size);
        std::memcpy(rel.pk, keys.data(), keys_size);
        rel.count = keys.size();

        // shuffle keys
        auto rng = std::default_random_engine {};
        std::shuffle(std::begin(keys), std::end(keys), rng);
        // TODO zipfian lookup patterns

//        cudaMallocManaged(&lookupKeys, keys_size);
        cudaMalloc(&lookupKeys, keys_size);
        cudaMemcpy(lookupKeys, keys.data(), keys_size, cudaMemcpyHostToDevice);/*
        cudaMemAdvise(lookupKeys, keys_size, cudaMemAdviseSetReadMostly, device_id);
        cudaMemPrefetchAsync(lookupKeys, keys_size, device_id);*/
    }

    btree::payload_t* tids;
    cudaMallocManaged(&tids, numElements*sizeof(decltype(tids)));

    int blockSize = 32;
    int numBlocks = (numElements + blockSize - 1) / blockSize;
    printf("numblocks: %d\n", numBlocks);

    printf("executing kernel...\n");
    auto startTs = std::chrono::high_resolution_clock::now();
    for (unsigned rep = 0; rep < maxRepetitions; ++rep) {
        rs_bulk_lookup<<<numBlocks, blockSize>>>(d_rs, numElements, lookupKeys, rel, tids);
        cudaDeviceSynchronize();
    }
    auto kernelStopTs = std::chrono::high_resolution_clock::now();
    auto kernelTime = chrono::duration_cast<chrono::microseconds>(kernelStopTs - startTs).count()/1000.;
    std::cout << "Kernel time: " << kernelTime << " ms\n";
    std::cout << "GPU MOps: " << (maxRepetitions*numElements/1e6)/(kernelTime/1e3) << endl;

    // validate results
    printf("validating results...\n");
    validate_results<<<numBlocks, blockSize>>>(tids, numElements, lookupKeys, rel);
    cudaDeviceSynchronize();

    return 0;
}
