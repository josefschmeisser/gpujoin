#pragma once

#include <cstdint>

#include "device_definitions.hpp"
#include "LinearProbingHashTable.cuh"

//using hj_ht_t = LinearProbingHashTable<indexed_t, size_t>;
using hj_ht_t = LinearProbingHashTable<index_key_t, device_size_t>;
using hj_device_ht_t = hj_ht_t::DeviceHandle;

struct hj_mutable_state {
    // Ephemeral state
    hj_device_ht_t ht;
};

struct hj_args {
    // Inputs
    const void* __restrict__ const build_side_rel;
    const device_size_t build_side_size;
    const void* __restrict__ const probe_side_rel;
    const device_size_t probe_side_size;
    // State and outputs
	hj_mutable_state* const state;
    value_t* __restrict__ tids;
};

template<class KeyType>
__global__ void hj_build_kernel(const hj_args args);

template<class KeyType>
__global__ void hj_probe_kernel(const hj_args args);
