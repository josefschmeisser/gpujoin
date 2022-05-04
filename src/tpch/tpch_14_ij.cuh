#pragma once

#include <cstdint>

#include "common.hpp"
#include "tpch_14_common.cuh"

struct join_entry {
    tid_t lineitem_tid;
    tid_t part_tid;
};

struct ij_mutable_state {
    // Ephemeral state
    numeric_raw_t* __restrict__ l_extendedprice_buffer = nullptr;
    numeric_raw_t* __restrict__ l_discount_buffer = nullptr;
    join_entry* __restrict__ join_entries = nullptr;
    unsigned output_index = 0u;
    // Cycle counters
    unsigned long long lookup_cycles = 0ull;
    unsigned long long scan_cycles = 0ull;
    unsigned long long sync_cycles = 0ull;
    unsigned long long sort_cycles = 0ull;
    unsigned long long join_cycles = 0ull;
    unsigned long long total_cycles = 0ull;
    // Outputs
    numeric_raw_t global_numerator = 0ll;
    numeric_raw_t global_denominator = 0ll;
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

// Pipelined Blockwise Sorting index join kernel
template<
    unsigned BLOCK_THREADS,
    unsigned ITEMS_PER_THREAD,
    class    IndexStructureType >
__launch_bounds__ (BLOCK_THREADS)
__global__ void ij_pbws (
    const ij_args args,
    const IndexStructureType index_structure
    );
