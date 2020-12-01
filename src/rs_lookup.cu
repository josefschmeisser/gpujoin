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

#include "rs/multi_map.h"
#include "btree.hpp"

using namespace std;

using rs_key_t = uint64_t;
using rs_rt_entry_t = uint32_t;
using rs_spline_point_t = rs::Coord<rs_key_t>;
using payload_t = btree::payload_t;

static constexpr unsigned numElements = 1e2;
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

template<typename T1, typename T2, typename P>
__device__ unsigned lower_bound(const T1& key, const T2* arr, const unsigned size, P cmp) {
    unsigned lower = 0;
    unsigned upper = size;
    do {
        unsigned mid = ((upper - lower) / 2) + lower;
        int c = cmp(arr[mid], key);
        if (c < 0) {
            upper = mid;
        } else if (c > 0) {
            lower = mid + 1;
        } else {
            return mid;
        }
    } while (lower < upper);
    return lower;
}

__global__ void do_lower_bound(const int* arr, const unsigned size) {
    lower_bound(0, arr, size, [] (const auto& a, const auto& b) {
        return a < b;
    });
}

/*
  // Returns the index of the spline point that marks the end of the spline segment that contains the `key`:
  // `key` ∈ (spline[index - 1], spline[index]]
  size_t GetSplineSegment(const KeyType key) const {
    // Narrow search range using radix table.
    const KeyType prefix = (key - min_key_) >> num_shift_bits_;
    assert (prefix + 1 < radix_table_.size());
    const uint32_t begin = radix_table_[prefix];
    const uint32_t end = radix_table_[prefix + 1];

    if (end - begin < 32) {
      // Do linear search over narrowed range.
      uint32_t current = begin;
      while (spline_points_[current].x < key) ++current;
      return current;
    }

    // Do binary search over narrowed range.
    const auto lb = std::lower_bound(spline_points_.begin() + begin,
                                     spline_points_.begin() + end,
                                     key,
                                     [](const Coord<KeyType>& coord, const KeyType key) { return coord.x < key; });
    return std::distance(spline_points_.begin(), lb);
  }
*/
__device__ unsigned get_spline_segment(ManagedRadixSpline* rs, const rs_key_t key) {
    printf("get_spline_segment: %lu\n", key);

    const rs_key_t prefix = (key - rs->min_key_) >> rs->num_shift_bits_;
    printf("key: %lu prefix: %lu\n", key, prefix);

    const uint32_t begin = rs->radix_table_[prefix];
    const uint32_t end = rs->radix_table_[prefix + 1];

    // TODO measure linear search for narrow ranges as in the reference implementation


    const auto range_size = end - begin;
    const auto lb = begin + lower_bound(key, rs->spline_points_ + begin, range_size, [] (const rs_spline_point_t& coord, const rs_key_t key) {
        return coord.x < key;
    });
    printf("key: %lu, lb: %u\n", key, lb);
    return lb;
}

/*
  // Returns the estimated position of `key`.
  double GetEstimatedPosition(const KeyType key) const {
    // Truncate to data boundaries.
    if (key <= min_key_) return 0;
    if (key >= max_key_) return num_keys_ - 1;

    // Find spline segment with `key` ∈ (spline[index - 1], spline[index]].
    const size_t index = GetSplineSegment(key);
    const Coord<KeyType> down = spline_points_[index - 1];
    const Coord<KeyType> up = spline_points_[index];

    // Compute slope.
    const double x_diff = up.x - down.x;
    const double y_diff = up.y - down.y;
    const double slope = y_diff / x_diff;

    // Interpolate.
    const double key_diff = key - down.x;
    return std::fma(key_diff, slope, down.y);
  }
*/

__device__ double get_estimate(ManagedRadixSpline* rs, const rs_key_t key) {
//TODO check: these two statements should not be required
    if (key <= rs->min_key_) return 0;
    if (key >= rs->max_key_) return rs->num_keys_ - 1;

    // find spline segment
    const unsigned index = get_spline_segment(rs, key);
printf("index: %u\n", index);
    const rs_spline_point_t& down = rs->spline_points_[index - 1];
    const rs_spline_point_t& up = rs->spline_points_[index];
printf("point: %f\n", down.x);
    // slope
    const double x_diff = up.x - down.x;
    const double y_diff = up.y - down.y;
    const double slope = y_diff / x_diff;

    // interpolate
    const double key_diff = key - down.x;
    return key_diff * slope + down.y;
}

/*
  // Returns a search bound [begin, end) around the estimated position.
  SearchBound GetSearchBound(const KeyType key) const {
    const size_t estimate = GetEstimatedPosition(key);
    const size_t begin = (estimate < max_error_) ? 0 : (estimate - max_error_);
    // `end` is exclusive.
    const size_t end = (estimate + max_error_ + 2 > num_keys_) ? num_keys_ : (estimate + max_error_ + 2);
    return SearchBound{begin, end};
  }
*/
__device__ payload_t rs_lookup(ManagedRadixSpline* rs, const rs_key_t key, const Relation& rel) {
    printf("key: %lu\n", key);
    const unsigned estimate = get_estimate(rs, key);
    printf("key: %lu estimate: %u\n", key, estimate);
    const unsigned begin = (estimate < rs->max_error_) ? 0 : (estimate - rs->max_error_);
    const unsigned end = (estimate + rs->max_error_ + 2 > rs->num_keys_) ? rs->num_keys_ : (estimate + rs->max_error_ + 2);
    printf("search bound [%u, %u)\n", begin, end);


    const auto bound_size = end - begin;
    const unsigned pos = lower_bound(key, rel.pk + begin, bound_size, [] (const rs_key_t& a, const rs_key_t& b) {
        return a < b;
    });

    printf("key: %lu pos: %u\n", pos);
    return (pos < rel.count) ? reinterpret_cast<payload_t>(pos) : invalidTid;
}

__global__ void rs_bulk_lookup(ManagedRadixSpline* rs, unsigned n, rs_key_t* keys, Relation rel, payload_t* tids) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    printf("index %d\n", index);
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

    /*
        // Search using RadixSpline.
        rs::SearchBound bound = rs.GetSearchBound(8128);
        cout << "The search key is in the range: ["
            << bound.begin << ", " << bound.end << ")" << endl;
        auto start = begin(keys) + bound.begin, last = begin(keys) + bound.end;
        cout << "The key is at position: " << std::lower_bound(start, last, 8128) - begin(keys) << endl;
    */

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

        cudaMalloc(&lookupKeys, numElements*sizeof(rs_key_t));
        // TODO shuffle keys/Zipfian lookup patterns
        cudaMemcpy(lookupKeys, keys.data(), numElements*sizeof(rs_key_t), cudaMemcpyHostToDevice);
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

    return 0;
}
