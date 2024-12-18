#pragma once

#include <cstdint>

#include "device_definitions.hpp"
#include "index_lookup_config.hpp"

template<class KeyType, class MapType>
struct hj_warpcore_args {
    using key_type = KeyType;
    using map_type = MapType;

    // Inputs
    const KeyType* __restrict__ const build_side_rel;
    const device_size_t build_side_size;
    const KeyType* __restrict__ const probe_side_rel;
    const device_size_t probe_side_size;
    MapType map;
    // Outputs
    value_t* __restrict__ tids;
};

template<class ArgsType >
__global__ void hj_warpcore_build_kernel(ArgsType args);

template<class ArgsType>
__global__ void hj_warpcore_probe_kernel(const ArgsType args);

#include "generic_hj_warpcore.inl"
