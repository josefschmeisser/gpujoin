#pragma once

#include "common.hpp"

#include "tpch_14_common.cuh"

#include <prefix_scan_state.h>

#include <gpu_radix_partition.h>


struct materialized_tuple {
    std::remove_pointer<decltype(lineitem_table_plain_t::l_extendedprice)>::type summand;
    std::remove_pointer<decltype(lineitem_table_plain_t::l_partkey)>::type l_partkey;
};

/*
struct materialization_buffer {
    decltype(lineitem_table_plain_t::l_extendedprice) __restrict__ summand;
    decltype(lineitem_table_plain_t::l_partkey) __restrict__ l_partkey;
};*/

#if 0
struct partitioned_index_join_args {
    // Inputs
    const lineitem_table_plain_t lineitem;
    const size_t lineitem_size;
    const part_table_plain_t part;
    //const void* index_structure;

    std::size_t const canonical_chunk_length; // TODO needed?
    uint32_t const padding_length;
    uint32_t const radix_bits;
    uint32_t const ignore_bits;
    // State
    /*
    std::tuple<
        decltype(lineitem.l_extendedprice),
        decltype(lineitem.l_discount),
        decltype(lineitem.l_partkey)
        >* __restrict__ materialized;*/
    //materialized_tuple* __restrict__ materialized;
	materialization_buffer materialized;
    uint32_t* materialized_size;

    ScanState<unsigned long long> *const prefix_scan_state;
    unsigned long long *const __restrict__ tmp_partition_offsets;
	unsigned long long *const __restrict__ partition_offsets;


    // Outputs
    int64_t* global_numerator;
    int64_t* global_denominator;
};
#endif

struct partitioned_index_join_mutable_state {
	// State
    decltype(lineitem_table_plain_t::l_partkey) const __restrict__ l_partkey;
    decltype(lineitem_table_plain_t::l_extendedprice) const __restrict__ summand;
    uint32_t materialized_size;
    // Outputs
    int64_t global_numerator;
    int64_t global_denominator;
};

struct partitioned_index_join_args {
    // Inputs
    const lineitem_table_plain_t lineitem;
    const size_t lineitem_size;
    const part_table_plain_t part;
    // State and outputs
	partitioned_index_join_mutable_state* const state;
};
