#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <chrono>
#include <cstring>

#include "rs/multi_map.h"

using namespace std;

using rs_key_t = uint64_t;
using rs_rt_entry_t = uint32_t;
using rs_spline_point_t = rs::Coord<rs_key_t>;

static constexpr unsigned numElements = 1e6;

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

int main(int argc, char** argv) {
    // Create random keys.
    vector<rs_key_t> keys(numElements);
    generate(keys.begin(), keys.end(), rand);
    keys.push_back(8128);
    sort(keys.begin(), keys.end());

    auto rs = builRadixSpline(keys);

    // Search using RadixSpline.
    rs::SearchBound bound = rs.GetSearchBound(8128);
    cout << "The search key is in the range: ["
        << bound.begin << ", " << bound.end << ")" << endl;
    auto start = begin(keys) + bound.begin, last = begin(keys) + bound.end;
    cout << "The key is at position: " << std::lower_bound(start, last, 8128) - begin(keys) << endl;

    RawRadixSpline* rrs = reinterpret_cast<RawRadixSpline*>(&rs);
    ManagedRadixSpline* mrs;
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

    return 0;
}
