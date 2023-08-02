#pragma once

#include <cstdint>
#include <memory>

#include "cuda_allocator.hpp"
#include "numa_allocator.hpp"
#include "mmap_allocator.hpp"

// TODO check crashes
//#define ONLY_AGGREGATES

enum class dataset_type : unsigned { dense, sparse };

enum class lookup_pattern_type : unsigned { uniform, uniform_unique, zipf };

// experiment config with default constants
struct experiment_config {
    std::string approach = "plain";
    std::string index_type = "btree";
    std::string scenario = "single";
    std::string output_file = "";
    uint64_t num_lookups = 1e8;
    uint64_t num_elements = 1e6;
    unsigned max_bits = 0;
    double zipf_factor = 1.25;
    int block_size = 128;
    dataset_type dataset = dataset_type::sparse;
    lookup_pattern_type lookup_pattern = lookup_pattern_type::uniform;
    bool sorted_lookups = false;
    unsigned partitioning_approach_ignore_bits = 4;
    bool partitioning_approach_dynamic_bit_range = true;
};

// constants:
static const uint32_t radix_bits = 11;

// types:
/*
using index_key_t = uint32_t;
using value_t = uint32_t;
*/

using index_key_t = uint64_t;
using value_t = uint64_t;

// allocators:

// host allocator
//template<class T> using host_allocator_t = huge_page_allocator<T>;
//template<class T> using host_allocator_t = mmap_allocator<T, huge_2mb, 0>;
template<class T> using host_allocator_t = std::allocator<T>;
//template<class T> using host_allocator_t = cuda_allocator<T, true>;

// device allocators
//template<class T> using device_index_allocator = cuda_allocator<T, cuda_allocation_type::device>;
template<class T> using device_index_allocator = cuda_allocator<T, cuda_allocation_type::zero_copy>;
//template<class T> using device_index_allocator = mmap_allocator<T, huge_2mb, 0>;

//using indexed_allocator_t = cuda_allocator<index_key_t, cuda_allocation_type::device>;
using indexed_allocator_t = cuda_allocator<index_key_t, cuda_allocation_type::zero_copy>;
//using indexed_allocator_t = mmap_allocator<index_key_t, huge_2mb, 0>;

//using lookup_keys_allocator_t = cuda_allocator<index_key_t, cuda_allocation_type::device>;
using lookup_keys_allocator_t = cuda_allocator<index_key_t, cuda_allocation_type::zero_copy>;
//using indexed_allocator_t = mmap_allocator<index_key_t, huge_2mb, 0>;

template<class T> using device_exclusive_allocator = cuda_allocator<T, cuda_allocation_type::device>;

experiment_config& get_experiment_config();

void parse_options(int argc, char** argv);
