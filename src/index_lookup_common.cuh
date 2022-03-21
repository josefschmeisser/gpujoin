#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>

#include "utils.hpp"
#include "zipf.hpp"

#include "cuda_utils.cuh"
#include "cuda_allocator.hpp"
#include "numa_allocator.hpp"
#include "mmap_allocator.hpp"
#include "indexes.cuh"
#include "device_array.hpp"


enum class dataset_type : unsigned { dense, uniform };

enum class lookup_pattern_type : unsigned { uniform, zipf };

template<class KeyType, class VectorType>
void generate_datasets(dataset_type dt, unsigned max_bits, VectorType& keys, lookup_pattern_type lookup_pattern, double zipf_factor, VectorType& lookups) {
    const std::size_t upper_limit = 1ul << (max_bits - 1u);
    auto rng = std::default_random_engine {};

    if (keys.size() - 1 > upper_limit) {
        throw std::runtime_error("resulting dataset would exceed the provided limit defined by 'max_bits'");
    }

    // generate dataset to be indexed
    if (dt == dataset_type::dense) {
        std::iota(keys.begin(), keys.end(), 0);
    } else if (dt == dataset_type::uniform) {
        // create random keys
        std::uniform_int_distribution<> key_distrib(0, upper_limit);
        std::unordered_set<KeyType> unique;
        unique.reserve(keys.size());
        while (unique.size() < keys.size()) {
            const auto key = key_distrib(rng);
            unique.insert(key);
        }

        std::copy(unique.begin(), unique.end(), keys.begin());
        std::sort(keys.begin(), keys.end());
    } else {
        assert(false);
    }

    // generate lookup keys
    if (lookup_pattern == lookup_pattern_type::uniform) {
        std::uniform_int_distribution<size_t> lookup_distribution(0, keys.size() - 1);
        std::generate(lookups.begin(), lookups.end(), [&]() { return keys[lookup_distribution(rng)]; });
    } else if (lookup_pattern == lookup_pattern_type::zipf) {
        std::mt19937 generator;
        generator.seed(0);
        zipf_distribution<uint64_t> lookup_distribution(keys.size() - 1, zipf_factor);
        for (uint64_t i = 0; i < lookups.size(); ++i) {
            const auto key_pos = lookup_distribution(generator);
            lookups[i] = keys[key_pos];
        }
    } else {
        assert(false);
    }

    //std::sort(lookups.begin(), lookups.end());
}

template<class KeyType, class IndexStructureType, class VectorType>
std::unique_ptr<IndexStructureType> build_index(const VectorType& h_keys, KeyType* d_keys) {
    auto index = std::make_unique<IndexStructureType>();
    index->construct(h_keys, d_keys);
    printf("index size: %lu bytes\n", index->memory_consumption());
    return index;
}

template<class IndexStructureType>
__global__ void lookup_kernel(const IndexStructureType index_structure, unsigned n, const index_key_t* __restrict__ keys, value_t* __restrict__ tids) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int i = index;
    uint32_t active_lanes = __ballot_sync(FULL_MASK, i < n);
    while (active_lanes) {
        bool active = i < n;
        auto tid = index_structure.cooperative_lookup(active, keys[i]);
        if (active) {
            tids[i] = tid;
        }

        i += stride;
        active_lanes = __ballot_sync(FULL_MASK, i < n);
    }
}

template<
    unsigned BLOCK_THREADS,
    unsigned ITEMS_PER_THREAD,
    class    IndexStructureType>
__global__ void lookup_kernel_with_sorting_v1(const IndexStructureType index_structure, unsigned n, const index_key_t* __restrict__ keys, value_t* __restrict__ tids) {
    enum { ITEMS_PER_ITERATION = BLOCK_THREADS*ITEMS_PER_THREAD };

    // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
    typedef cub::BlockLoad<index_key_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoad;

    //typedef cub::BlockRadixSort<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;
    using BlockRadixSortT = cub::BlockRadixSort<index_key_t, BLOCK_THREADS, ITEMS_PER_THREAD, uint32_t>;

    __shared__ union TempStorage {
        // Allocate shared memory for BlockLoad
        typename BlockLoad::TempStorage load;

        typename BlockRadixSortT::TempStorage sort;
    } temp_storage;

    __shared__ index_key_t buffer[ITEMS_PER_ITERATION];
    __shared__ uint32_t in_buffer_pos[ITEMS_PER_ITERATION];
    __shared__ uint32_t buffer_idx;

    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    const unsigned tile_size = min(n, (n + gridDim.x - 1) / gridDim.x);
    unsigned tid = blockIdx.x * tile_size; // first tid where cub::BlockLoad starts scanning (has to be the same for all threads in this block)
    const unsigned tid_limit = min(tid + tile_size, n);

//if (lane_id == 0) printf("warp: %d tile_size: %d\n", warp_id, tile_size);

    const unsigned iteration_count = (tile_size + ITEMS_PER_ITERATION - 1) / ITEMS_PER_ITERATION;

    index_key_t input_thread_data[ITEMS_PER_THREAD]; // TODO omit this


    for (int i = 0; i < iteration_count; ++i) {
//if (lane_id == 0) printf("warp: %d iteration: %d first tid: %d\n", warp_id, i, tid);

        unsigned valid_items = min(ITEMS_PER_ITERATION, tid_limit - tid);
//if (lane_id == 0) printf("warp: %d valid_items: %d\n", warp_id, valid_items);

        // Load a segment of consecutive items that are blocked across threads
        BlockLoad(temp_storage.load).Load(keys + tid, input_thread_data, valid_items);

        __syncthreads();

        // reset shared memory variables
        if (lane_id == 0) {
            buffer_idx = 0;
        }

        #pragma unroll
        for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
            const unsigned pos = threadIdx.x*ITEMS_PER_THREAD + j;
            buffer[pos] = input_thread_data[j];
            in_buffer_pos[pos] = tid + pos;
        }

        __syncthreads();

#if 1
        // we only perform the sort step when the buffer is completely filled
        if (valid_items == ITEMS_PER_ITERATION) {
            using key_array_t = index_key_t[ITEMS_PER_THREAD];
            using value_array_t = uint32_t[ITEMS_PER_THREAD];

            index_key_t* thread_keys_raw = &buffer[threadIdx.x*ITEMS_PER_THREAD];
            uint32_t* thread_values_raw = &in_buffer_pos[threadIdx.x*ITEMS_PER_THREAD];
            key_array_t& thread_keys = reinterpret_cast<key_array_t&>(*thread_keys_raw);
            value_array_t& thread_values = reinterpret_cast<value_array_t&>(*thread_values_raw);

            BlockRadixSortT(temp_storage.sort).Sort(thread_keys, thread_values, 4, max_bits);

             __syncthreads();
        }/* else {
//if (lane_id == 0) printf("warp: %d iteration: %d - skipping sort step ===\n", warp_id, i);
        }*/
#endif

        // empty buffer
        unsigned old;
        do {
            if (lane_id == 0) {
                old = atomic_add_sat(&buffer_idx, 32u, valid_items);
            }
            old = __shfl_sync(FULL_MASK, old, 0);
            unsigned actual_count = min(valid_items - old, 32);
//if (lane_id == 0) printf("warp: %d iteration: %d - actual_count: %u\n", warp_id, i, actual_count);

            if (actual_count == 0) break;

            bool active = lane_id < actual_count;

            uint32_t assoc_tid = 0;
            index_key_t element;
            if (active) {
                assoc_tid = in_buffer_pos[old + lane_id];
                element = buffer[old + lane_id];
//printf("warp: %d lane: %d - tid: %u element: %u\n", warp_id, lane_id, assoc_tid, element);
            }

            value_t tid_b = index_structure.cooperative_lookup(active, element);
            if (active) {
//printf("warp: %d lane: %d - tid_b: %u\n", warp_id, lane_id, tid_b);
                tids[assoc_tid] = tid_b;
            }

//printf("warp: %d lane: $d - element: %u\n", warp_id, lane_id, );

        } while (true);

        tid += valid_items;
    }
}

std::vector<std::pair<std::string, std::string>> allocator_type_names() {
    std::vector<std::pair<std::string, std::string>> r = {
        std::make_pair(std::string("host_allocator"), std::string(type_name<host_allocator_t<int>>::value())),
        std::make_pair(std::string("device_index_allocator"), std::string(type_name<device_index_allocator<int>>::value())),
        std::make_pair(std::string("indexed_allocator"), std::string(type_name<indexed_allocator_t>::value())),
        std::make_pair(std::string("lookup_keys_allocator"), std::string(type_name<lookup_keys_allocator_t>::value()))
    };
    return r;
}
