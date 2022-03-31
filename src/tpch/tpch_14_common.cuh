#pragma once

#include "common.hpp"
#include "config.hpp"
#include "indexes.cuh"

// allocators:
template<class T> using host_allocator = HOST_ALLOCATOR_TYPE;
template<class T> using device_index_allocator = DEVICE_INDEX_ALLOCATOR_TYPE;
template<class T> using device_table_allocator = DEVICE_RELATION_ALLOCATOR;

using indexed_t = std::remove_pointer_t<decltype(lineitem_table_plain_t::l_partkey)>;
using payload_t = uint32_t;

// index structure types:
using btree_type = btree_index<indexed_t, payload_t, device_index_allocator, host_allocator>;
using harmonia_type = harmonia_index<indexed_t, payload_t, device_index_allocator, host_allocator>;
using lower_bound_type = lower_bound_index<indexed_t, payload_t, device_index_allocator, host_allocator>;
using radix_spline_type = radix_spline_index<indexed_t, payload_t, device_index_allocator, host_allocator>;
using no_op_type = no_op_index<indexed_t, payload_t, device_index_allocator, host_allocator>;

void execute_approach(std::string approach);
