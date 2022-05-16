#pragma once

#include <cstdint>

#include "common.hpp"
#include "tpch_14_common.cuh"
#include "LinearProbingHashTable.cuh"

using hj_ht_t = LinearProbingHashTable<indexed_t, size_t>;
using hj_device_ht_t = hj_ht_t::DeviceHandle;

struct hj_mutable_state {
    // Ephemeral state
    hj_device_ht_t ht;
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
    // State and outputs
	hj_mutable_state* const state;
};

__global__ void hj_build_kernel(hj_args args);

__global__ void hj_probe_kernel(hj_args args);
