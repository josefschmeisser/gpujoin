#pragma once

#include <cstdint>

#include "device_definitions.hpp"
#include "linear_probing_hashtable.cuh"
#include "index_lookup_config.hpp"

template<class KeyType, class MapViewType, class MapMutableViewType>
struct hj_cuco_args {
    // Inputs
    const KeyType* __restrict__ const build_side_rel;
    const device_size_t build_side_size;
    const KeyType* __restrict__ const probe_side_rel;
    const device_size_t probe_side_size;
    MapViewType map_view;
    MapMutableViewType map_mutable_view;
    // Outputs
    value_t* __restrict__ tids;
};

template<uint32_t block_size, uint32_t cg_size, class ArgsType>
__global__ void hj_cuco_build_kernel(ArgsType args);

template<class ArgsType>
__global__ void hj_cuco_probe_kernel(const ArgsType args);

#include "generic_hj_cuco.inl"
