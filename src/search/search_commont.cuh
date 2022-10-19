#pragma once

#include <cstdint>
#include "common.hpp"
#include "config.hpp"
#include "indexes.cuh"

#include "mmap_allocator.hpp"

// allocators:
template<class T> using host_allocator = HOST_ALLOCATOR_TYPE;
template<class T> using device_table_allocator = DEVICE_RELATION_ALLOCATOR;
template<class T> using device_exclusive_allocator = cuda_allocator<uint8_t, cuda_allocation_type::device>;

using value_t = uint64_t;

void generate_data();

void execute_approach(std::string approach);
