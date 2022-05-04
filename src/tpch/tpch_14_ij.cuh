#pragma once

#include "tpch_14_common.cuh"

#include <cstdint>

#include "common.hpp"

struct join_entry {
    uint32_t lineitem_tid;
    uint32_t part_tid;
};

struct ij_mutable_state {
    // Ephemeral state
    int64_t* __restrict__ l_extendedprice_buffer,
    int64_t* __restrict__ l_discount_buffer
    join_entry* __restrict__ join_entries
    uint32_t __restrict__ output_index;
    // Cycle counters
    unsigned long long lookup_cycles;
    unsigned long long scan_cycles;
    unsigned long long sync_cycles;
    unsigned long long sort_cycles;
    unsigned long long join_cycles;
    unsigned long long total_cycles;
    // Outputs
    int64_t global_numerator;
    int64_t global_denominator;
};

struct ij_args {
    // Inputs
    const lineitem_table_plain_t* const lineitem;
    const size_t lineitem_size;
    const part_table_plain_t* const part;
    const size_t part_size;
    // State and outputs
	ij_mutable_state* const state;
};

/*
struct partitioned_ij_lookup_args {
    // Inputs
    const part_table_plain_t* const part;
	void* rel;
    uint32_t rel_length;
    uint32_t rel_padding_length;
    unsigned long long* rel_partition_offsets;
    uint32_t* task_assignment;
    uint32_t radix_bits;
    // State and outputs
	partitioned_ij_lookup_mutable_state* const state;
};
*/

template<
    unsigned BLOCK_THREADS,
    unsigned ITEMS_PER_THREAD,
    class    IndexStructureType >
__launch_bounds__ (BLOCK_THREADS)
__global__ void ij_pbws (
    const lineitem_table_plain_t* __restrict__ lineitem,
    const unsigned lineitem_size,
    const part_table_plain_t* __restrict__ part,
    const unsigned part_size,
    const IndexStructureType index_structure,
    int64_t* __restrict__ l_extendedprice_buffer,
    int64_t* __restrict__ l_discount_buffer
    );

