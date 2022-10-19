#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <limits>

// experiment config with default constants
struct experiment_config {
    std::string approach = "plain";
    uint64_t num_lookups = 1e8;
    uint64_t num_elements = 1e6;
    unsigned max_bits = 24;
    int block_size = 128;
    dataset_type dataset = dataset_type::sparse;
    lookup_pattern_type lookup_pattern = lookup_pattern_type::uniform;
    bool sorted_lookups = false;
};

// allocators:

// host allocator
//#define HOST_ALLOCATOR_TYPE std::allocator<T>
//#define HOST_ALLOCATOR_TYPE cuda_allocator<T, cuda_allocation_type::zero_copy>
#define HOST_ALLOCATOR_TYPE mmap_allocator<T, huge_2mb, 0>
//#define HOST_ALLOCATOR_TYPE mmap_allocator<T, huge_2mb, 1>

// device relation allocator
//#define DEVICE_RELATION_ALLOCATOR cuda_allocator<T, cuda_allocation_type::zero_copy>
#define DEVICE_RELATION_ALLOCATOR mmap_allocator<T, huge_2mb, 0>

experiment_config& get_experiment_config();

void parse_options(int argc, char** argv);
