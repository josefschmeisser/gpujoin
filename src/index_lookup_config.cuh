#pragma once

#include <cstdint>
#include <memory>

#include "cuda_allocator.hpp"
#include "numa_allocator.hpp"
#include "mmap_allocator.hpp"
#include "indexes.cuh"

// standard constants
static const unsigned default_num_lookups = 1e8;
static const unsigned default_num_elements = 1e6;
static const unsigned max_bits = 24;
static const bool partitial_sorting = false;

// types
using index_key_t = uint32_t;
using value_t = uint32_t;

// allocators:

// host allocator
//template<class T> using host_allocator_t = huge_page_allocator<T>;
//template<class T> using host_allocator_t = mmap_allocator<T, huge_2mb, 1>;
template<class T> using host_allocator_t = std::allocator<T>;
//template<class T> using host_allocator_t = cuda_allocator<T, true>;

// device allocators
//template<class T> using device_index_allocator = cuda_allocator<T, cuda_allocation_type::device>;
template<class T> using device_index_allocator = cuda_allocator<T, cuda_allocation_type::zero_copy>;
//using indexed_allocator_t = cuda_allocator<index_key_t>;
using indexed_allocator_t = cuda_allocator<index_key_t, cuda_allocation_type::zero_copy>;
using lookup_keys_allocator_t = cuda_allocator<index_key_t>;

//using index_type = lower_bound_index<index_key_t, value_t, device_index_allocator, host_allocator_t>;
//using index_type = harmonia_index<index_key_t, value_t, device_index_allocator, host_allocator_t>;
//using index_type = btree_index<index_key_t, value_t, device_index_allocator, host_allocator_t>;
using index_type = radix_spline_index<index_key_t, value_t, device_index_allocator, host_allocator_t>;
