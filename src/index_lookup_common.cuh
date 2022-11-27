#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>

#include <oneapi/tbb/parallel_for.h>

#include "utils.hpp"
#include "zipf.hpp"

#include "cuda_utils.cuh"
#include "device_properties.hpp"
#include "cuda_allocator.hpp"
#include "numa_allocator.hpp"
#include "mmap_allocator.hpp"
#include "indexes.cuh"
#include "device_array.hpp"
#include "index_lookup_config.hpp"
#include "index_lookup_config.tpp"

using namespace oneapi::tbb;

using btree_type = btree_index<index_key_t, value_t, device_index_allocator, host_allocator_t>;
using harmonia_type = harmonia_index<index_key_t, value_t, device_index_allocator, host_allocator_t>;
using binary_search_type = binary_search_index<index_key_t, value_t, device_index_allocator, host_allocator_t>;
using radix_spline_type = radix_spline_index<index_key_t, value_t, device_index_allocator, host_allocator_t>;
using no_op_type = no_op_index<index_key_t, value_t, device_index_allocator, host_allocator_t>;


template<class T, class VectorType>
void populate_densely(VectorType& v) {
    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, v.size()), [&](const oneapi::tbb::blocked_range<size_t>& r) {
        std::iota(v.begin() + r.begin(), v.begin() + r.end(), r.begin());
    });
}

template<class T, class VectorType>
void populate_uniformly(VectorType& v, uint64_t limit) {
    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<uint64_t>(0ul, v.size()), [&](const oneapi::tbb::blocked_range<uint64_t>& r) {
        std::mt19937 rng {r.begin()};
        std::uniform_int_distribution<T> lookup_distribution(0ul, limit - 1ul);
        std::generate(v.begin() + r.begin(), v.begin() + r.end(), [&]() { return lookup_distribution(rng); });
        /*
        for (uint64_t i = r.begin(); i < r.end(); ++i) {
            const auto key_idx = lookup_distribution(rng);
            assert(key_idx < v.size());
            v[i] = key_idx;
        }*/
    });
}

template<class T, class VectorType>
void populate_uniquely_uniformly(VectorType& v, unsigned max_bits) {
    if (max_bits < std::log2(2*v.size())) {
        // increase sparsity by providing a larger value for max_bits
        throw std::runtime_error("not sparse enough");
    }
    const std::size_t upper_limit = 1ul << (max_bits - 1u);

    // TODO: consider https://stackoverflow.com/a/6953958

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, v.size()), [&](const oneapi::tbb::blocked_range<size_t>& r) {
        //printf("range begin: %lu end: %lu\n", r.begin(), r.end());

        // create random keys
        std::mt19937 rng {r.begin()};
        std::uniform_int_distribution<T> key_distrib(0, upper_limit);
        std::unordered_set<T> unique;
        unique.reserve(v.size());

        while (unique.size() < v.size()) {
            const auto key = key_distrib(rng);
            unique.insert(key);
        }

        std::copy(unique.begin(), unique.end(), v.begin() + r.begin());
        std::sort(v.begin() + r.begin(), v.begin() + r.end());
    });
}

template<class T, class VectorType>
void populate_with_zipfian_pattern(VectorType& v, uint64_t limit, double zipf_factor) {
    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<uint64_t>(0ul, v.size()), [&](const oneapi::tbb::blocked_range<uint64_t>& r) {
        std::mt19937 rng {r.begin()};
        zipf_distribution<T> lookup_distribution(limit - 1ul, zipf_factor);
        for (uint64_t i = 0; i < r.size(); ++i) {
            const auto key_pos = lookup_distribution(rng);
            v[r.begin() + i] = key_pos;
        }
    });
}

template<class KeyType, class VectorType>
void generate_datasets(dataset_type dt, unsigned max_bits, VectorType& keys, lookup_pattern_type lookup_pattern, double zipf_factor, VectorType& lookups) {
    const std::size_t upper_limit = (max_bits > 0) ? (1ul << (max_bits - 1u)) : std::numeric_limits<KeyType>::max();

    if (keys.size() - 1 > upper_limit) {
        throw std::runtime_error("resulting dataset would exceed the provided limit defined by 'max_bits'");
    }

    // generate dataset to be indexed
    printf("generating dataset to be indexed...\n");
    switch (dt) {
        case dataset_type::dense:
            populate_densely<KeyType, VectorType>(keys);
            break;
        case dataset_type::sparse:
            populate_uniquely_uniformly<KeyType, VectorType>(keys, max_bits);
            break;
        default:
            assert(false);
    }

    // generate lookup keys
    printf("generating lookups...\n");
    switch (lookup_pattern) {
        case lookup_pattern_type::uniform:
            populate_uniformly<uint64_t, VectorType>(lookups, keys.size());
            break;
        case lookup_pattern_type::zipf:
            populate_with_zipfian_pattern<uint64_t, VectorType>(lookups, keys.size(), zipf_factor);
            break;
        default:
            assert(false);
    }

    // map key positions into keys
    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<uint64_t>(0ul, lookups.size()), [&](const oneapi::tbb::blocked_range<uint64_t>& r) {
        for (uint64_t i = 0; i < r.size(); ++i) {
            assert(lookups[r.begin() + i < keys.size()]);
            lookups[r.begin() + i] = keys[lookups[r.begin() + i]];
        }
    });
}

template<class KeyType, class IndexStructureType, class VectorType>
std::unique_ptr<IndexStructureType> build_index(const VectorType& h_keys, KeyType* d_keys) {
    auto index = std::make_unique<IndexStructureType>();
    const auto view = make_vector_view(const_cast<VectorType&>(h_keys));
    index->construct(view, d_keys);
    printf("index size: %lu bytes\n", index->memory_consumption());
    return index;
}

template<class IndexStructureType>
__global__ void lookup_kernel(const IndexStructureType index_structure, unsigned n, const index_key_t* __restrict__ keys, value_t* __restrict__ tids) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
#if 0
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
#else
    const int loop_limit = (n + warpSize - 1) & ~(warpSize - 1); // round to next multiple of warpSize
    for (int i = index; i < loop_limit; i += stride) {
        const bool active = i < n;
        const auto tid = index_structure.cooperative_lookup(active, keys[i]);
        if (active) {
            tids[i] = tid;
        }
    }
#endif
}

template<
    unsigned BLOCK_THREADS,
    unsigned ITEMS_PER_THREAD,
    class    IndexStructureType>
__global__ void lookup_kernel_with_sorting_v1(const IndexStructureType index_structure, unsigned n, const index_key_t* __restrict__ keys, value_t* __restrict__ tids, unsigned max_bits) {
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
    (void)warp_id;

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
                //old = atomic_add_sat(&buffer_idx, 32u, valid_items);
                old = atomicAdd(&buffer_idx, 32u);
            }
            old = __shfl_sync(FULL_MASK, old, 0);
            //unsigned actual_count = min(valid_items - old, 32);
            unsigned actual_count = (old > valid_items) ? 0 : valid_items - old;
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

template<class IndexType>
std::vector<std::pair<std::string, std::string>> create_common_experiment_description_pairs() {
    const auto& config = get_experiment_config();

    std::vector<std::pair<std::string, std::string>> r = {
        std::make_pair(std::string("device"), std::string(get_device_properties(0).name)),
        std::make_pair(std::string("index_type"), std::string(type_name<IndexType>::value())),
        std::make_pair(std::string("dataset"), tmpl_to_string(config.dataset)),
        std::make_pair(std::string("lookup_pattern"), tmpl_to_string(config.lookup_pattern)),
        std::make_pair(std::string("num_elements"), std::to_string(config.num_elements)),
        std::make_pair(std::string("num_lookups"), std::to_string(config.num_lookups)),
        // allocators:
        std::make_pair(std::string("host_allocator"), std::string(type_name<host_allocator_t<int>>::value())),
        std::make_pair(std::string("device_index_allocator"), std::string(type_name<device_index_allocator<int>>::value())),
        std::make_pair(std::string("indexed_allocator"), std::string(type_name<indexed_allocator_t>::value())),
        std::make_pair(std::string("lookup_keys_allocator"), std::string(type_name<lookup_keys_allocator_t>::value()))
    };

    if (config.dataset == dataset_type::sparse) {
        r.push_back(std::make_pair(std::string("max_bits"), std::to_string(config.max_bits)));
    }

    if (config.lookup_pattern == lookup_pattern_type::zipf) {
        r.push_back(std::make_pair(std::string("zipf_factor"), std::to_string(config.zipf_factor)));
    }

    return r;
}
