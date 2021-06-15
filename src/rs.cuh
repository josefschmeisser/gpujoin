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

namespace rs {

using rs_rt_entry_t = uint32_t;

template<class Key>
struct RawRadixSpline {
    using key_t = Key;
    using spline_point_t = rs::Coord<key_t>;

    key_t min_key_;
    key_t max_key_;
    size_t num_keys_;
    size_t num_radix_bits_;
    size_t num_shift_bits_;
    size_t max_error_;

    std::vector<rs_rt_entry_t> radix_table_;
    std::vector<spline_point_t> spline_points_;
};

template<class Key>
struct DeviceRadixSpline {
    using key_t = Key;
    using spline_point_t = rs::Coord<key_t>;

    key_t min_key_;
    key_t max_key_;
    size_t num_keys_;
    size_t num_radix_bits_;
    size_t num_shift_bits_;
    size_t max_error_;

    rs_rt_entry_t* radix_table_;
    spline_point_t* spline_points_;
};

template<class Key>
auto build_radix_spline(const std::vector<Key>& keys) {
    auto min = keys.front();
    auto max = keys.back();
    rs::Builder<Key> rsb(min, max);
    for (const auto& key : keys) rsb.AddKey(key);
    rs::RadixSpline<Key> rs = rsb.Finalize();
    return rs;
}

template<class Policy, class Key>
DeviceRadixSpline<Key>* copy_radix_spline(const rs::RadixSpline<Key>& radixSpline) {
    static Policy f;

    static DeviceRadixSpline<Key> tmp;
    const RawRadixSpline<Key>* rrs = reinterpret_cast<const RawRadixSpline<Key>*>(&radixSpline);
    std::memcpy(&tmp, &radixSpline, sizeof(DeviceRadixSpline<Key>));

    // copy radix table
    tmp.radix_table_ = f(rrs->radix_table_);

    // copy spline points
    tmp.spline_points_ = f(rrs->spline_points_);

    DeviceRadixSpline<Key>* d_rs;
    cudaMalloc(&d_rs, sizeof(DeviceRadixSpline<Key>));
    cudaMemcpy(d_rs, &tmp, sizeof(DeviceRadixSpline<Key>), cudaMemcpyHostToDevice);
    return d_rs;
}

namespace cuda {

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

template<class Key>
__device__ unsigned get_spline_segment(const DeviceRadixSpline<Key>* rs, const Key key) {
    const auto prefix = (key - rs->min_key_) >> rs->num_shift_bits_;

    const uint32_t begin = rs->radix_table_[prefix];
    const uint32_t end = rs->radix_table_[prefix + 1];

    // TODO measure linear search for narrow ranges as in the reference implementation

    const auto range_size = end - begin;
    const auto lb = begin + lower_bound(key, rs->spline_points_ + begin, range_size, [] (const auto& coord, const Key key) {
        return coord.x < key;
    });
//    printf("key: %lu, lb: %u\n", key, lb);
    return lb;
}

template<class Key>
__device__ double get_estimate(const DeviceRadixSpline<Key>* rs, const Key key) {
    if (key <= rs->min_key_) return 0;
    if (key >= rs->max_key_) return rs->num_keys_ - 1;

    // find spline segment
    const unsigned index = get_spline_segment(rs, key);
    const auto& down = rs->spline_points_[index - 1];
    const auto& up = rs->spline_points_[index];

    // slope
    const double x_diff = up.x - down.x;
    const double y_diff = up.y - down.y;
    const double slope = y_diff / x_diff;

    // interpolate
    const double key_diff = key - down.x;
    return key_diff * slope + down.y;
}

}

}
