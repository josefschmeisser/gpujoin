#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <bits/c++config.h>
#include <bits/stdint-uintn.h>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <chrono>
#include <cstring>
#include <stdexcept>

#include "rs/multi_map.h"
#include "btree.hpp"

using namespace std;

using rs_key_t = uint64_t;
using rs_rt_entry_t = uint32_t;
using rs_spline_point_t = rs::Coord<rs_key_t>;
using payload_t = btree::payload_t;

static constexpr int device_id = 0;
static constexpr unsigned numElements = 1e8;
static constexpr payload_t invalidTid = std::numeric_limits<payload_t>::max();

struct Relation {
    size_t count;
    rs_key_t* pk;
    uint64_t* payload;
};

struct RawRadixSpline {
    rs_key_t min_key_;
    rs_key_t max_key_;
    size_t num_keys_;
    size_t num_radix_bits_;
    size_t num_shift_bits_;
    size_t max_error_;

    std::vector<rs_rt_entry_t> radix_table_;
    std::vector<rs::Coord<rs_key_t>> spline_points_;
};

struct ManagedRadixSpline {
    rs_key_t min_key_;
    rs_key_t max_key_;
    size_t num_keys_;
    size_t num_radix_bits_;
    size_t num_shift_bits_;
    size_t max_error_;

    rs_rt_entry_t* radix_table_;
    rs_spline_point_t* spline_points_;
};

auto builRadixSpline(const vector<rs_key_t>& keys) {
    auto min = keys.front();
    auto max = keys.back();
    rs::Builder<rs_key_t> rsb(min, max);
    for (const auto& key : keys) rsb.AddKey(key);
    rs::RadixSpline<rs_key_t> rs = rsb.Finalize();
    return rs;
}

namespace gpu {

template<typename T, typename P>
__device__ unsigned lower_bound(const T& key, const T* arr, const unsigned size) {
    unsigned lower = 0;
    unsigned upper = size;
    do {
        unsigned mid = ((upper - lower) / 2) + lower;
        int c = cmp(arr[mid], key); // a < b
        if (key < arr[mid]) {
            upper = mid;
        } else if (key > arr[mid]) {
            lower = mid + 1;
        } else {
            return mid;
        }
    } while (lower < upper);
    return lower;
}

template<typename T1, typename T2, typename P>
__device__ unsigned lower_bound(const T1& key, const T2* arr, const unsigned size, P cmp) {
    unsigned lower = 0;
    unsigned count = size;
    while (count > 0) {
        unsigned step = count / 2;
        unsigned mid = lower + step;
        if (cmp(arr[mid], key)) {
            lower = mid + 1;
            count -= step + 1;
        } else {
            count = step;
        }
    }
    return lower;
}

__global__ void do_lower_bound(const int* arr, const unsigned size) {
    lower_bound(0, arr, size, [] (const auto& a, const auto& b) {
        return a < b;
    });
}

__device__ unsigned get_spline_segment(ManagedRadixSpline* rs, const rs_key_t key) {
    const rs_key_t prefix = (key - rs->min_key_) >> rs->num_shift_bits_;

    const uint32_t begin = rs->radix_table_[prefix];
    const uint32_t end = rs->radix_table_[prefix + 1];

    // TODO measure linear search for narrow ranges as in the reference implementation

    const auto range_size = end - begin;
    const auto lb = begin + lower_bound(key, rs->spline_points_ + begin, range_size, [] (const rs_spline_point_t& coord, const rs_key_t key) {
        return coord.x < key;
    });
//    printf("key: %lu, lb: %u\n", key, lb);
    return lb;
}

__device__ double get_estimate(ManagedRadixSpline* rs, const rs_key_t key) {
    if (key <= rs->min_key_) return 0;
    if (key >= rs->max_key_) return rs->num_keys_ - 1;

    // find spline segment
    const unsigned index = get_spline_segment(rs, key);
    const rs_spline_point_t& down = rs->spline_points_[index - 1];
    const rs_spline_point_t& up = rs->spline_points_[index];

    // slope
    const double x_diff = up.x - down.x;
    const double y_diff = up.y - down.y;
    const double slope = y_diff / x_diff;

    // interpolate
    const double key_diff = key - down.x;
    return key_diff * slope + down.y;
}

__device__ payload_t rs_lookup(ManagedRadixSpline* rs, const rs_key_t key, const Relation& rel) {
    const unsigned estimate = get_estimate(rs, key);
//    printf("key: %lu estimate: %u\n", key, estimate);
    const unsigned begin = (estimate < rs->max_error_) ? 0 : (estimate - rs->max_error_);
    const unsigned end = (estimate + rs->max_error_ + 2 > rs->num_keys_) ? rs->num_keys_ : (estimate + rs->max_error_ + 2);

    const auto bound_size = end - begin;
    const unsigned pos = begin + lower_bound(key, &rel.pk[begin], bound_size, [] (const rs_key_t& a, const rs_key_t& b) -> int {
        return a < b;
    });
//    int index = blockIdx.x * blockDim.x + threadIdx.x;
//    printf("key: %lu search bound [%u, %u) pos: %u expected: %d\n", key, begin, end, pos, index);
    return (pos < rel.count) ? reinterpret_cast<payload_t>(pos) : invalidTid;
}

__global__ void rs_bulk_lookup(ManagedRadixSpline* rs, unsigned n, const rs_key_t* keys, Relation rel, payload_t* tids) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        tids[i] = rs_lookup(rs, keys[i], rel);
    }
}

};

int main(int argc, char** argv) {

    Relation rel;
    rs_key_t* lookupKeys;
    ManagedRadixSpline* mrs;

    {
        // Create random keys.
        vector<rs_key_t> keys(numElements - 1);
        generate(keys.begin(), keys.end(), rand);
        keys.push_back(8128);
        sort(keys.begin(), keys.end());

        auto rs = builRadixSpline(keys);

        RawRadixSpline* rrs = reinterpret_cast<RawRadixSpline*>(&rs);
        cudaMallocManaged(&mrs, sizeof(ManagedRadixSpline));
        std::memcpy(mrs, &rs, sizeof(ManagedRadixSpline));
        // copy radix table
        const auto rs_table_size = sizeof(rs_rt_entry_t)*rrs->radix_table_.size();
        cudaMallocManaged(&mrs->radix_table_, rs_table_size);
        std::memcpy(mrs->radix_table_, rrs->radix_table_.data(), rs_table_size);
        // copy spline points
        const auto rs_spline_points_size = sizeof(rs_spline_point_t)*rrs->spline_points_.size();
        cudaMallocManaged(&mrs->spline_points_, rs_spline_points_size);
        std::memcpy(mrs->spline_points_, rrs->spline_points_.data(), rs_spline_points_size);

        const auto keys_size = sizeof(rs_key_t)*keys.size();
        cudaMallocManaged(&rel.pk, keys_size);
        std::memcpy(rel.pk, keys.data(), keys_size);
        rel.count = keys.size();

        printf("radix table size: %lu\n", rrs->radix_table_.size());

        cudaMallocManaged(&lookupKeys, keys_size);
        // TODO shuffle keys/Zipfian lookup patterns
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
    gpu::rs_bulk_lookup<<<numBlocks, blockSize>>>(mrs, numElements, lookupKeys, rel, tids);
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
