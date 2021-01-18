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

static constexpr int device_id = 0;
static constexpr unsigned numElements = 1e8;
static constexpr payload_t invalidTid = std::numeric_limits<payload_t>::max();

struct Relation {
    size_t count;
    rs_key_t* pk;
    uint64_t* payload;
};

__device__ payload_t rs_lookup(DeviceRadixSpline* rs, const rs_key_t key, const Relation& rel) {
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

__global__ void rs_bulk_lookup(DeviceRadixSpline* rs, unsigned n, const rs_key_t* keys, Relation rel, payload_t* tids) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        tids[i] = rs_lookup(rs, keys[i], rel);
    }
}

int main(int argc, char** argv) {

    Relation rel;
    rs_key_t* lookupKeys;
    DeviceRadixSpline* d_rs;

    {
        // Create random keys.
        vector<rs_key_t> keys(numElements - 1);
        generate(keys.begin(), keys.end(), rand);
        keys.push_back(8128);
        sort(keys.begin(), keys.end());

        auto rs = build_radix_spline(keys);
/*
        RawRadixSpline* rrs = reinterpret_cast<RawRadixSpline*>(&rs);
        cudaMallocManaged(&d_rs, sizeof(DeviceRadixSpline));
        std::memcpy(d_rs, &rs, sizeof(DeviceRadixSpline));
        // copy radix table
        const auto rs_table_size = sizeof(rs_rt_entry_t)*rrs->radix_table_.size();
        cudaMallocManaged(&d_rs->radix_table_, rs_table_size);
        std::memcpy(d_rs->radix_table_, rrs->radix_table_.data(), rs_table_size);
        // copy spline points
        const auto rs_spline_points_size = sizeof(rs_spline_point_t)*rrs->spline_points_.size();
        cudaMallocManaged(&d_rs->spline_points_, rs_spline_points_size);
        std::memcpy(d_rs->spline_points_, rrs->spline_points_.data(), rs_spline_points_size);
*/
        d_rs = rs::copy_radix_spline<vector_to_managed_array>(rs);

        const auto keys_size = sizeof(rs_key_t)*keys.size();
        cudaMallocManaged(&rel.pk, keys_size);
        std::memcpy(rel.pk, keys.data(), keys_size);
        rel.count = keys.size();

//        printf("radix table size: %lu\n", rrs->radix_table_.size());

        // shuffle keys
        auto rng = std::default_random_engine {};
        std::shuffle(std::begin(keys), std::end(keys), rng);
        // TODO zipfian lookup patterns

        cudaMallocManaged(&lookupKeys, keys_size);
        cudaMemcpy(lookupKeys, keys.data(), keys_size, cudaMemcpyHostToDevice);
        cudaMemAdvise(lookupKeys, keys_size, cudaMemAdviseSetReadMostly, device_id);
        cudaMemPrefetchAsync(lookupKeys, keys_size, device_id);
    }

    btree::payload_t* tids;
    cudaMallocManaged(&tids, numElements*sizeof(decltype(tids)));

    int blockSize = 32;
    int numBlocks = (numElements + blockSize - 1) / blockSize;
    printf("numblocks: %d\n", numBlocks);

    auto startTs = std::chrono::high_resolution_clock::now();
    rs_bulk_lookup<<<numBlocks, blockSize>>>(d_rs, numElements, lookupKeys, rel, tids);
    cudaDeviceSynchronize();
    auto kernelStopTs = std::chrono::high_resolution_clock::now();
    auto kernelTime = chrono::duration_cast<chrono::microseconds>(kernelStopTs - startTs).count()/1000.;
    std::cout << "Kernel time: " << kernelTime << " ms\n";
    std::cout << "GPU MOps: " << (numElements/1e6)/(kernelTime/1e3) << endl;

    for (unsigned i = 0; i < numElements; ++i) {
        size_t pos = reinterpret_cast<size_t>(tids[i]);
        if (lookupKeys[i] != rel.pk[pos]) throw std::runtime_error("key mismatch");
    }

    return 0;
}
