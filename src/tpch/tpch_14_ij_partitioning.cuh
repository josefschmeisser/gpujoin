#pragma once

#include <fast-interconnects/gpu_common.h>
#include <fast-interconnects/gpu_radix_partition.h>
#include <fast-interconnects/prefix_scan_state.h>

#include "common.hpp"
#include "tpch_14_common.cuh"

struct materialized_tuple {
    std::remove_pointer<decltype(lineitem_table_plain_t::l_partkey)>::type l_partkey;
    std::remove_pointer<decltype(lineitem_table_plain_t::l_extendedprice)>::type summand;
};

// Note: we use 64 bit keys here because the payload also has 64 bit and both types are required to have the same width
using partitioning_indexed_t = int64_t;
static_assert(sizeof(partitioning_indexed_t) >= sizeof(decltype(materialized_tuple::l_partkey)));

using partitioning_payload_t = int64_t;
static_assert(sizeof(partitioning_payload_t) >= sizeof(decltype(materialized_tuple::summand)));

using partitioned_tuple_type = Tuple<partitioning_indexed_t, partitioning_payload_t>;

struct partitioned_ij_scan_mutable_state {
	// State
    /*
    decltype(lineitem_table_plain_t::l_partkey) const __restrict__ l_partkey;
    decltype(lineitem_table_plain_t::l_extendedprice) const __restrict__ summand;
    */
    partitioning_indexed_t* const l_partkey;
    partitioning_payload_t* const summand;
    uint32_t materialized_size;
};

struct partitioned_ij_scan_args {
    // Inputs
    const lineitem_table_plain_t* const lineitem;
    const size_t lineitem_size;
    // State and outputs
	partitioned_ij_scan_mutable_state* const state;
};

struct partitioned_ij_lookup_mutable_state {
    // Outputs
    int64_t global_numerator;
    int64_t global_denominator;
};

struct partitioned_ij_lookup_args {
    // Inputs
    const part_table_plain_t* const part;
	void* rel;
    uint32_t rel_length;
    uint32_t rel_padding_length;
    unsigned long long* rel_partition_offsets;
    uint32_t* task_assignments;
    uint32_t radix_bits;
    // State and outputs
	partitioned_ij_lookup_mutable_state* const state;
};

__global__ void partitioned_ij_scan(partitioned_ij_scan_args args);

__global__ void partitioned_ij_scan_refill(partitioned_ij_scan_args args);

template<class IndexStructureType>
__global__ void partitioned_ij_lookup(const partitioned_ij_lookup_args args, const IndexStructureType index_structure);
