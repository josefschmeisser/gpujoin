#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <chrono>

#include "rs/multi_map.h"

using namespace std;

using key_t = uint64_t;

static constexpr unsigned numElements = 1e6;

struct ManagedRadixSpline {
    key_t min_key_;
    key_t max_key_;
    size_t num_keys_;
    size_t num_radix_bits_;
    size_t num_shift_bits_;
    size_t max_error_;
/*
    std::vector<uint32_t> radix_table_;
    std::vector<rs::Coord<KeyType>> spline_points_;*/
    uint32_t radix_table_;
    rs::Coord<key_t>* spline_points_;
};

auto builRadixSpline(const vector<key_t>& keys) {
    auto min = keys.front();
    auto max = keys.back();
    rs::Builder<key_t> rsb(min, max);
    for (const auto& key : keys) rsb.AddKey(key);
    rs::RadixSpline<key_t> rs = rsb.Finalize();
    return rs;
}

int main(int argc, char** argv) {
    // Create random keys.
    vector<key_t> keys(numElements);
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



    return 0;
}
