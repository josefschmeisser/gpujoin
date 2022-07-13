#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <limits>

// experiment config with default constants
struct experiment_config {
    std::string approach = "hj";
    std::string db_path = "n/a";
    std::string index_type = "btree";
    bool prefetch_index = false;
    bool sort_indexed_relation = true;
    int block_size = 128;
};

// allocators:

// host allocator
#define HOST_ALLOCATOR_TYPE std::allocator<T>
//#define HOST_ALLOCATOR_TYPE cuda_allocator<T, cuda_allocation_type::zero_copy>
//#define HOST_ALLOCATOR_TYPE mmap_allocator<T, huge_2mb, 0>
//#define HOST_ALLOCATOR_TYPE mmap_allocator<T, huge_2mb, 1>

// device index allocator
//#define DEVICE_INDEX_ALLOCATOR_TYPE cuda_allocator<T, cuda_allocation_type::zero_copy>
#define DEVICE_INDEX_ALLOCATOR_TYPE cuda_allocator<T, cuda_allocation_type::device>
//#define DEVICE_INDEX_ALLOCATOR_TYPE mmap_allocator<T, huge_2mb, 0>

// device relation allocator
#define DEVICE_RELATION_ALLOCATOR cuda_allocator<T, cuda_allocation_type::zero_copy>
//#define DEVICE_RELATION_ALLOCATOR mmap_allocator<T, huge_2mb, 0>

static const uint32_t lower_shipdate = 2449962; // 1995-09-01
static const uint32_t upper_shipdate = 2449992; // 1995-10-01
static const uint32_t invalid_tid __attribute__((unused)) = std::numeric_limits<uint32_t>::max();

experiment_config& get_experiment_config();

void parse_options(int argc, char** argv);
