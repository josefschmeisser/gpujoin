#pragma once

#include <cstdint>

#include "common.hpp"
#include "device_definitions.hpp"
#include "tpch_14_common.cuh"
#include "linear_probing_hashtable.cuh"

using hj_ht_t = linear_probing_hashtable<indexed_t, device_size_t>;
using hj_device_ht_t = hj_ht_t::device_handle;

struct hj_mutable_state {
    // Outputs
    numeric_raw_t global_numerator = 0ll;
    numeric_raw_t global_denominator = 0ll;
};

struct hj_args {
    // Inputs
    const lineitem_table_plain_t* const lineitem;
    const size_t lineitem_size;
    const part_table_plain_t* const part;
    const size_t part_size;
    const hj_device_ht_t ht;
    // State and outputs
	hj_mutable_state* const state;
};

__global__ void hj_build_kernel(hj_args args);

__global__ void hj_probe_kernel(hj_args args);
