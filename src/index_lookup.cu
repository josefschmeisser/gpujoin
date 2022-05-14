#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <chrono>
#include <memory>
#include <unordered_set>

#include <cuda_runtime.h>
#include <sys/types.h>

#include "tpch/common.hpp"
#include "utils.hpp"
#include "zipf.hpp"
#include "cuda_utils.cuh"
#include "device_array.hpp"
#include "measuring.hpp"
#include "device_properties.hpp"
#include "indexes.cuh"

#include "index_lookup_config.hpp"
#include "index_lookup_common.cuh"

using namespace std;
using namespace measuring;

static const int blockSize = 64;
//static const int blockSize = 256; // best for sorting on pascal
static const unsigned repetitions = 10;
static const unsigned activeLanes = 32;

using index_type = INDEX_TYPE;

/*
std::string get_cuda_device_name(unsigned device_id) {
    char device_name[256];
    const auto error_id = cuDeviceGetName(device_name, 256, device_id);
    if (error_id != CUDA_SUCCESS) {
        std::cerr << "failed to fetch device name for device: " << device_id << std::endl;
        return "unknown";
    }
    return device_name;
}*/

static experiment_description create_experiment_description() {
    const auto& config = get_experiment_config();

    experiment_description r;
    r.name = "plain_lookup";
    r.approach = config.partitial_sorting ? "partial_sorting" : "plain";
    r.other = create_common_experiment_description_pairs<index_type>();
    return r;
}

#if 0
template<
    unsigned BLOCK_THREADS,
    unsigned ITEMS_PER_THREAD,
    class    IndexStructureType>
__global__ void lookup_kernel_with_sorting_v2(const IndexStructureType index_structure, unsigned n, const index_key_t* __restrict__ keys, value_t* __restrict__ tids) {
    enum { ITEMS_PER_ITERATION = BLOCK_THREADS*ITEMS_PER_THREAD };

    // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
    typedef cub::BlockLoad<index_key_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoad;

    typedef cub::BlockRadixSort<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;

    __shared__ union TempStorage {
        // Allocate shared memory for BlockLoad
        typename BlockLoad::TempStorage load;

        typename BlockRadixSortT::TempStorage sort;
    } temp_storage;

    __shared__ uint64_t buffer[ITEMS_PER_ITERATION];
    __shared__ uint32_t buffer_idx;

    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

//    const unsigned tile_size = round_up_pow2((n + BLOCK_THREADS - 1) / gridDim.x); // TODO cache-line allignment should be sufficient
    const unsigned tile_size = min(n, (n + BLOCK_THREADS - 1) / gridDim.x);
    unsigned tid = blockIdx.x * tile_size; // first tid where cub::BlockLoad starts scanning (has to be the same for all threads in this block)
    const unsigned tid_limit = min(tid + tile_size, n);

//if (lane_id == 0) printf("warp: %d tile_size: %d\n", warp_id, tile_size);

    const unsigned iteration_count = (tile_size + ITEMS_PER_ITERATION - 1) / ITEMS_PER_ITERATION;


    using key_value_array_t = uint64_t[ITEMS_PER_THREAD];
    uint64_t* thread_data_raw = &buffer[threadIdx.x*ITEMS_PER_THREAD];
    key_value_array_t& thread_data = reinterpret_cast<key_value_array_t&>(*thread_data_raw);


    for (int i = 0; i < iteration_count; ++i) {
//if (lane_id == 0) printf("warp: %d iteration: %d first tid: %d\n", warp_id, i, tid);

        unsigned valid_items = min(ITEMS_PER_ITERATION, n - tid);
//if (lane_id == 0) printf("warp: %d valid_items: %d\n", warp_id, valid_items);


        // Load a segment of consecutive items that are blocked across threads
        BlockLoad(temp_storage.load).Load(keys + tid, thread_data, valid_items);

        __syncthreads();

        // reset shared memory variables
        if (lane_id == 0) {
            buffer_idx = 0;
        }

        #pragma unroll
        for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
            uint64_t upper = tid + threadIdx.x*ITEMS_PER_THREAD + j;
            buffer[threadIdx.x*ITEMS_PER_THREAD + j] = upper<<32 | static_cast<uint64_t>(input_thread_data[j]);
        }


        __syncthreads();

#if 0
        // we only perform the sort step when the buffer is completely filled
        if (valid_items == ITEMS_PER_ITERATION) {
//if (lane_id == 0) printf("warp: %d iteration: %d - sorting... ===\n", warp_id, i);
            BlockRadixSortT(temp_storage.sort).Sort(thread_data, 4, max_bits); // TODO
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
            index_key_t element = 0xffffffff;
            if (active) {
                assoc_tid = buffer[old + lane_id] >> 32;
                element = buffer[old + lane_id] & 0xffffffff;
//printf("warp: %d lane: %d - tid: %u element: %u\n", warp_id, lane_id, assoc_tid, element);
            }

            value_t tid_b = index_structure.cooperative_lookup(active, element);
            if (active) {
//printf("warp: %d lane: %d - tid_b: %u\n", warp_id, lane_id, tid_b);
                tids[assoc_tid] = tid_b;
            }

//printf("warp: %d lane: $d - element: %u\n", warp_id, lane_id, );

        } while (true);//actual_count == 32);


        tid += valid_items;
    }
}
#endif


#if 0
// TODO port
void run_lane_limited_lookup_benchmark() {
    // calculate a factory by which the number of lookups has to be scaled in order to be able to serve all threads
    int lookupFactor = 32/activeLanes + ((32%activeLanes > 0) ? 1 : 0);
    const int numAugmentedLookups = numLookups*lookupFactor;
    std::cout << "lookup scale factor: " << lookupFactor << " numAugmentedLookups: " << numAugmentedLookups << std::endl;


    const int threadCount = ((numAugmentedLookups + 31) & (-32));
    std::cout << "n: " << numAugmentedLookups << " threadCount: " << threadCount << std::endl;
    decltype(std::chrono::high_resolution_clock::now()) kernelStart;


    std::cout << "active lanes: " << activeLanes << std::endl;
    kernelStart = std::chrono::high_resolution_clock::now();
    for (unsigned rep = 0; rep < maxRepetitions; ++rep) {
        btree_bulk_lookup_serialized<activeLanes><<<numBlocks, blockSize>>>(d_tree, numAugmentedLookups, d_lookupKeys, d_tids);
        cudaDeviceSynchronize();
    }

    const auto kernelStop = std::chrono::high_resolution_clock::now();
    const auto kernelTime = chrono::duration_cast<chrono::microseconds>(kernelStop - kernelStart).count()/1000.;
    std::cout << "Kernel time: " << kernelTime << " ms\n";
    std::cout << "GPU MOps: " << (maxRepetitions*numLookups/1e6)/(kernelTime/1e3) << endl;
}
#endif
