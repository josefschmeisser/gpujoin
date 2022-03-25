#pragma once

#include <cstdint>
#include <memory>

#include "cuda_allocator.hpp"
#include "numa_allocator.hpp"
#include "mmap_allocator.hpp"

enum class dataset_type : unsigned { dense, sparse };

enum class lookup_pattern_type : unsigned { uniform, zipf };

// experiment config with default constants
struct experiment_config {
    unsigned num_lookups = 1e8;
    unsigned num_elements = 1e6;
    unsigned max_bits = 24;
    double zipf_factor = 1.25;
    bool partitial_sorting = false;
    dataset_type dataset = dataset_type::sparse;
    lookup_pattern_type lookup_pattern = lookup_pattern_type::uniform;
};

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

#define INDEX_TYPE btree_index<index_key_t, value_t, device_index_allocator, host_allocator_t>;
//#define INDEX_TYPE lower_bound_index<index_key_t, value_t, device_index_allocator, host_allocator_t>;
//#define INDEX_TYPE harmonia_index<index_key_t, value_t, device_index_allocator, host_allocator_t>;
//#define INDEX_TYPE radix_spline_index<index_key_t, value_t, device_index_allocator, host_allocator_t>;

experiment_config& get_experiment_config();

void parse_options(int argc, char** argv);
