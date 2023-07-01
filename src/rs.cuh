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
#include "device_array.hpp"
#include "search.cuh"
#include "utils.hpp"

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

template<class Key, bool IsConst>
//struct DeviceRadixSpline {
struct device_radix_spline_tmpl {
    using key_t = Key;
    using spline_point_t = rs::Coord<key_t>;

    add_const_if_t<key_t, IsConst> min_key_ = { };
    add_const_if_t<key_t, IsConst> max_key_ = { };
    add_const_if_t<size_t, IsConst> num_keys_ = 0;
    add_const_if_t<size_t, IsConst> num_radix_bits_ = 0;
    add_const_if_t<size_t, IsConst> num_shift_bits_ = 0;
    add_const_if_t<size_t, IsConst> max_error_ = 0;

    add_const_if_t<const rs_rt_entry_t* __restrict__, IsConst> radix_table_ = nullptr;
    add_const_if_t<const spline_point_t* __restrict__, IsConst> spline_points_ = nullptr;
};

template<class Key>
using DeviceRadixSpline = device_radix_spline_tmpl<Key, false>;

template<class Key>
using device_radix_spline = device_radix_spline_tmpl<Key, false>;

template<class Key>
using const_device_radix_spline = device_radix_spline_tmpl<Key, true>;

template<class Vector>
auto build_radix_spline(const Vector& keys) {
    using key_type = typename Vector::value_type;

    auto min = keys.front();
    auto max = keys.back();
    rs::Builder<key_type> rsb(min, max);
    for (const auto& key : keys) rsb.AddKey(key);
    rs::RadixSpline<key_type> rs = rsb.Finalize();
    return rs;
}

#if 0
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
#endif

template<class Key>
struct device_array_guard {
    using key_t = Key;

    device_array_wrapper<rs_rt_entry_t> radix_table_guard;
    device_array_wrapper<typename DeviceRadixSpline<key_t>::spline_point_t> spline_points_guard;
};

template<class Key, class TargetAllocator>
auto migrate_radix_spline(rs::RadixSpline<Key>& rs, DeviceRadixSpline<Key>& d_rs, TargetAllocator& target_allocator) {
    device_array_guard<Key> guard;

    auto rrs = reinterpret_cast<RawRadixSpline<Key>*>(&rs);
    std::memcpy(&d_rs, &rs, sizeof(DeviceRadixSpline<Key>));

    // copy radix table
    typename TargetAllocator::rebind<rs_rt_entry_t>::other table_entry_allocator = target_allocator;
    guard.radix_table_guard = create_device_array_from(rrs->radix_table_, table_entry_allocator);
    d_rs.radix_table_ = guard.radix_table_guard.data();

    // copy spline points
    typename TargetAllocator::rebind<typename DeviceRadixSpline<Key>::spline_point_t>::other spline_point_allocator = target_allocator;
    guard.spline_points_guard = create_device_array_from(rrs->spline_points_, spline_point_allocator);
    d_rs.spline_points_ = guard.spline_points_guard.data();

    return guard;
}

namespace cuda {

template<class Key>
__device__ device_size_t get_spline_segment(const const_device_radix_spline<Key>& rs, const Key key) {
    const auto prefix = (key - rs.min_key_) >> rs.num_shift_bits_;

    const rs_rt_entry_t begin = rs.radix_table_[prefix];
    const rs_rt_entry_t end = rs.radix_table_[prefix + 1];

    // TODO measure linear search for narrow ranges as in the reference implementation

    const auto range_size = end - begin;
    const auto lb = begin + lower_bound(key, rs.spline_points_ + begin, range_size, [] (const auto& coord, const Key key) {
        return coord.x < key;
    });

    //printf("key: %lu, lb: %u\n", key, lb);
    return lb;
}

template<class Key>
__device__ double get_estimate(const const_device_radix_spline<Key>& rs, const Key key) {
    if (key <= rs.min_key_) return 0;
    if (key >= rs.max_key_) return rs.num_keys_ - 1;

    // find spline segment
    const auto index = get_spline_segment(rs, key);
    //const auto& down = rs.spline_points_[index - 1];
    const rs::Coord<Key> down = rs.spline_points_[index - 1];
    //const auto& up = rs.spline_points_[index];
    const rs::Coord<Key> up = rs.spline_points_[index];

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
