#include "common.hpp"
#include "config.hpp"

#if 0
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <device_atomic_functions.h>
#include <driver_types.h>
#include <iostream>
#include <iomanip>
#include <limits>
#include <cassert>
#include <cstring>
#include <chrono>
#include <sys/types.h>
#include <unordered_map>

#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <vector>

#include "cuda_utils.cuh"
#include "cuda_allocator.hpp"
#include "numa_allocator.hpp"
#include "mmap_allocator.hpp"
#endif
#include "indexes.cuh"
//#include "device_array.hpp"

#include "tpch_14_common.cuh"

//#include "prefix_scan_state.h"
#include <prefix_scan_state.h>

#include "cuda_utils.cuh"

/*
TODO
*/

__managed__ int64_t globalSum1 = 0;
__managed__ int64_t globalSum2 = 0;

#if 0
template<
    unsigned BLOCK_THREADS,
    unsigned ITEMS_PER_THREAD,
    class    IndexStructureType >
__launch_bounds__ (BLOCK_THREADS)
__global__ void ij_join_finalization_kernel(
    uint32_t* __restrict__ g_l_partkey_buffer,
    int64_t* __restrict__ g_l_extendedprice_buffer,
    int64_t* __restrict__ g_l_discount_buffer,
    const unsigned lineitem_buffer_size,
    const part_table_plain_t* __restrict__ part,
    const unsigned part_size,
    const IndexStructureType index_structure
    )
{
    enum { ITEMS_PER_ITERATION = BLOCK_THREADS*ITEMS_PER_THREAD };

    using index_key_t = uint32_t;

    using BlockLoadT = cub::BlockLoad<index_key_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockRadixSortT = cub::BlockRadixSort<index_key_t, BLOCK_THREADS, ITEMS_PER_THREAD, uint32_t>;

    __shared__ union TempStorage {
        // Allocate shared memory for BlockLoad
        typename BlockLoadT::TempStorage load;

        typename BlockRadixSortT::TempStorage sort;
    } temp_storage;

    __shared__ index_key_t buffer[ITEMS_PER_ITERATION];
    __shared__ uint32_t in_buffer_pos[ITEMS_PER_ITERATION];
    __shared__ uint32_t buffer_idx;

    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    const unsigned tile_size = min(lineitem_buffer_size, (lineitem_buffer_size + gridDim.x - 1) / gridDim.x);
    unsigned tid = blockIdx.x * tile_size; // first tid where cub::BlockLoad starts scanning (has to be the same for all threads in this block)
    const unsigned tid_limit = min(tid + tile_size, lineitem_buffer_size);


    const unsigned iteration_count = (tile_size + ITEMS_PER_ITERATION - 1) / ITEMS_PER_ITERATION;
//if (lane_id == 0) printf("warp: %d tile_size: %d iteration_count: %u\n", warp_id, tile_size, iteration_count);

    index_key_t input_thread_data[ITEMS_PER_THREAD]; // TODO omit this

    int64_t sum1 = 0;
    int64_t sum2 = 0;

    for (int i = 0; i < iteration_count; ++i) {
//if (lane_id == 0) printf("block: %d warp: %d iteration: %d first tid: %d\n", blockIdx.x, warp_id, i, tid);

//        unsigned valid_items = min(ITEMS_PER_ITERATION, lineitem_buffer_size - tid);
        unsigned valid_items = min(ITEMS_PER_ITERATION, tid_limit - tid);
//if (lane_id == 0) printf("warp: %d iteration: %d valid_items: %d\n", warp_id, i, valid_items);

        // Load a segment of consecutive items that are blocked across threads
        BlockLoadT(temp_storage.load).Load(g_l_partkey_buffer + tid, input_thread_data, valid_items);

        __syncthreads();

        // reset shared memory variables
        if (lane_id == 0) {
            buffer_idx = 0;
        }

        #pragma unroll
        for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
            const unsigned pos = threadIdx.x*ITEMS_PER_THREAD + j;
            buffer[pos] = g_l_partkey_buffer[tid + pos];
            in_buffer_pos[pos] = tid + pos;
        }

        __syncthreads();

#ifndef SKIP_SORT
        // we only perform the sort step when the buffer is completely filled
        if (valid_items == ITEMS_PER_ITERATION) {
            using key_array_t = index_key_t[ITEMS_PER_THREAD];
            using value_array_t = uint32_t[ITEMS_PER_THREAD];

            index_key_t* thread_keys_raw = &buffer[threadIdx.x*ITEMS_PER_THREAD];
            uint32_t* thread_values_raw = &in_buffer_pos[threadIdx.x*ITEMS_PER_THREAD];
            key_array_t& thread_keys = reinterpret_cast<key_array_t&>(*thread_keys_raw);
            value_array_t& thread_values = reinterpret_cast<value_array_t&>(*thread_values_raw);

            BlockRadixSortT(temp_storage.sort).Sort(thread_keys, thread_values, 4, 22);

             __syncthreads();
        }
#endif

        // empty buffer
        unsigned old;
        do {
            if (lane_id == 0) {
                old = atomic_add_sat(&buffer_idx, 32u, valid_items);
            }
            old = __shfl_sync(FULL_MASK, old, 0);
            unsigned actual_count = min(valid_items - old, 32);
//if (lane_id == 0) printf("warp: %d iteration: %d - old: %u actual_count: %u\n", warp_id, i, old, actual_count);

            if (actual_count == 0) break;

            bool active = lane_id < actual_count;

            uint32_t assoc_tid = 0;
            index_key_t l_partkey;
            if (active) {
//if (lane_id == 31) printf("warp: %d iteration: %d - access buffer pos: %u\n", warp_id, i, old + lane_id);
                assoc_tid = in_buffer_pos[old + lane_id];
                l_partkey = buffer[old + lane_id];
//printf("block: %d warp: %d lane: %d - tid: %u l_partkey: %u\n", blockIdx.x, warp_id, lane_id, assoc_tid, l_partkey);
            }

            payload_t tid_b = index_structure.cooperative_lookup(active, l_partkey);
            active = active && (tid_b != invalid_tid);

            sum2 += active;

            // evaluate predicate
            if (active) {
                const auto summand = g_l_extendedprice_buffer[assoc_tid] * (100 - g_l_discount_buffer[assoc_tid]);
                sum2 += summand;

                const char* type = reinterpret_cast<const char*>(&part->p_type[tid_b]); // FIXME relies on undefined behavior
                if (my_strcmp(type, "PROMO", 5) == 0) {
                    sum1 += summand;
                }
            }
        } while (true);

        tid += valid_items;
    }

//printf("sum1: %lu sum2: %lu\n", sum1, sum2);
    // reduce both sums
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum1 += __shfl_down_sync(FULL_MASK, sum1, offset);
        sum2 += __shfl_down_sync(FULL_MASK, sum2, offset);
    }
    if (lane_id == 0) {
        atomicAdd((unsigned long long int*)&globalSum1, (unsigned long long int)sum1);
        atomicAdd((unsigned long long int*)&globalSum2, (unsigned long long int)sum2);
    }
}

struct left_pipeline_result {
    size_t size;
    device_array_wrapper<uint32_t> l_partkey_buffer_guard;
    device_array_wrapper<int64_t> l_extendedprice_buffer_guard;
    device_array_wrapper<int64_t> l_discount_buffer_guard;
};

auto left_pipeline(Database& db) {
    auto& lineitem = db.lineitem;

    std::vector<uint32_t> l_partkey_buffer;
    std::vector<int64_t> l_extendedprice_buffer;
    std::vector<int64_t> l_discount_buffer;

    for (size_t i = 0; i < lineitem.l_partkey.size(); ++i) {
        if (lineitem.l_shipdate[i].raw < lower_shipdate ||
            lineitem.l_shipdate[i].raw >= upper_shipdate) {
            continue;
        }

        l_partkey_buffer.push_back(lineitem.l_partkey[i]);
        l_extendedprice_buffer.push_back(lineitem.l_extendedprice[i].raw);
        l_discount_buffer.push_back(lineitem.l_discount[i].raw);
    }

#ifdef PRE_SORT
    auto permutation = compute_permutation(l_partkey_buffer.begin(), l_partkey_buffer.end(), std::less<>{});
    apply_permutation(permutation, l_partkey_buffer, l_extendedprice_buffer, l_discount_buffer);
#endif

    // copy elements
    cuda_allocator<int, cuda_allocation_type::device> device_buffer_allocator;
    auto l_partkey_buffer_guard = create_device_array_from(l_partkey_buffer, device_buffer_allocator);
    auto l_extendedprice_buffer_guard = create_device_array_from(l_extendedprice_buffer, device_buffer_allocator);
    auto l_discount_buffer_guard = create_device_array_from(l_discount_buffer, device_buffer_allocator);

    return left_pipeline_result{l_partkey_buffer.size(), std::move(l_partkey_buffer_guard), std::move(l_extendedprice_buffer_guard), std::move(l_discount_buffer_guard)};
}
#endif

#if 0
void run_ij_buffer() {
    using namespace std;

    enum { BLOCK_THREADS = 256, ITEMS_PER_THREAD = 10 }; // TODO optimize

    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    int num_blocks = num_sms*4; // TODO

    auto left = left_pipeline(db);

    uint32_t* d_l_partkey_buffer = left.l_partkey_buffer_guard.data();
    int64_t* d_l_extendedprice_buffer = left.l_extendedprice_buffer_guard.data();
    int64_t* d_l_discount_buffer = left.l_discount_buffer_guard.data();

    const auto kernelStart = std::chrono::high_resolution_clock::now();

    ij_join_finalization_kernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(d_l_partkey_buffer, d_l_extendedprice_buffer, d_l_discount_buffer, left.size, part_device, part_size, index_structure.device_index);
    cudaDeviceSynchronize();

    const auto kernelStop = std::chrono::high_resolution_clock::now();
    const auto kernelTime = std::chrono::duration_cast<std::chrono::microseconds>(kernelStop - kernelStart).count()/1000.;
    std::cout << "kernel time: " << kernelTime << " ms\n";
}
#endif

#if 0
template<
    unsigned BLOCK_THREADS,
    unsigned ITEMS_PER_THREAD,
    class    IndexStructureType >
__launch_bounds__ (BLOCK_THREADS)
__device__ void ij_join_streamed(
/*
    uint32_t* __restrict__ g_l_partkey_buffer,
    int64_t* __restrict__ g_l_extendedprice_buffer,
    int64_t* __restrict__ g_l_discount_buffer,
    const unsigned lineitem_buffer_size,
    const part_table_plain_t* __restrict__ part,
    const unsigned part_size,*/
    const IndexStructureType index_structure

    )
{
    printf("from gpu\n");
}
#endif


// Exports the vanilla index join kernel for 8-byte keys.
extern "C" __launch_bounds__(1024, 1) __global__ void ij_join_streamed_btree(const btree_type::device_index_t index) {
    printf("from gpu\n");
}




template<class IndexStructureType>
__global__ void test_kernel() {
    printf("from gpu\n");
}

template __global__ void test_kernel<btree_type>();
template __global__ void test_kernel<harmonia_type>();
template __global__ void test_kernel<lower_bound_type>();
template __global__ void test_kernel<radix_spline_type>();
template __global__ void test_kernel<no_op_type>();


struct materialized_tuple {
    decltype(lineitem_table_plain_t::l_extendedprice) summand;
    //decltype(lineitem_table_plain_t::l_extendedprice) l_extendedprice;
    //decltype(lineitem_table_plain_t::l_discount) l_discount;
    decltype(lineitem_table_plain_t::l_partkey) l_partkey;
};

struct partitioned_index_join_args {
    // Input
    const lineitem_table_plain_t lineitem;
    const size_t lineitem_size;
    const part_table_plain_t part;
    const void* index_structure;

    std::size_t const canonical_chunk_length; // TODO needed?
    uint32_t const padding_length;
    uint32_t const radix_bits;
    uint32_t const ignore_bits;
    // State
    std::tuple<
        decltype(lineitem.l_extendedprice),
        decltype(lineitem.l_discount),
        decltype(lineitem.l_partkey)
        >* materialized;
    uint32_t* materialized_size;

    ScanState<unsigned long long> *const prefix_scan_state;
    unsigned long long *const __restrict__ tmp_partition_offsets;



    // Output
    int64_t* global_numerator;
    int64_t* global_denominator;
};


//template<class IndexStructureType>
//__global__ void partitioned_ij_scan(const lineitem_table_plain_t* __restrict__ lineitem, const unsigned lineitem_size, const part_table_plain_t* __restrict__ part, IndexStructureType index_structure) {
__global__ void partitioned_ij_scan(partitioned_index_join_args args) {

    const auto* __restrict__ l_shipdate = args.lineitem.l_shipdate;
    const auto* __restrict__ l_extendedprice = args.lineitem.l_extendedprice;
    const auto* __restrict__ l_discount = args.lineitem.l_discount;
    const auto* __restrict__ l_partkey = args.lineitem.l_partkey;

    const unsigned my_lane = lane_id();

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = index; i < args.lineitem_size + 31; i += stride) {
        bool active = i < args.lineitem_size;

        // evaluate predicate
        active = (l_shipdate[i] >= lower_shipdate) && (l_shipdate[i] < upper_shipdate);

        const auto summand = l_extendedprice[i] * (100 - l_discount[i]);
        const auto partkey = l_partkey[i];

        // TODO materialize

        // determine all threads with matching tuples
        uint32_t mask = __ballot_sync(FULL_MASK, active);
        const auto count = __popc(mask);
        if (count < 1) continue;

        // update global buffer index
        const uint32_t right = __funnelshift_l(FULL_MASK, 0, my_lane);
        const unsigned offset = __popc(mask & right);
        uint32_t base = 0;
        if (my_lane == 0) {
            base = atomicAdd(args.materialized_size, count);
        }
        base = __shfl_sync(FULL_MASK, base, 0);

        if (active) {
//            printf("lane %u store to: %u\n", my_lane, base + offset);
            auto& tuple = args.materialized[base + offset];
            std::get<0>(tuple) = summand; // FIXME
            std::get<1>(tuple) = partkey; // FIXME
        }
    }

    // TODO compute prefix sum
}

__global__ void partitioned_ij_scan_refill(partitioned_index_join_args args) {
    static constexpr unsigned tuples_per_thread = 2;
    //static constexpr unsigned per_warp_buffer_size = 32 * (tuples_per_thread + 1);
    static constexpr unsigned per_warp_buffer_size = 32 * tuples_per_thread;

    extern __shared__ uint32_t shared_mem[];

    const auto* __restrict__ l_shipdate = args.lineitem.l_shipdate;
    const auto* __restrict__ l_extendedprice = args.lineitem.l_extendedprice;
    const auto* __restrict__ l_discount = args.lineitem.l_discount;
    const auto* __restrict__ l_partkey = args.lineitem.l_partkey;

    const unsigned my_lane = lane_id();

    const unsigned warp_id = (threadIdx.x / 32);

    //static constexpr 
    //decltype(args.lineitem.l_partkey)* partkey_buffer = reinterpret_cast<decltype(args.lineitem.l_partkey)*>(shared_mem) + (warp_id * 32);
    materialized_tuple* tuple_buffer = reinterpret_cast<materialized_tuple*>(shared_mem) + (warp_id * per_warp_buffer_size);
    unsigned buffer_count = 0;

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = index; i < args.lineitem_size + 31; i += stride) {
        bool active = i < args.lineitem_size;

        // evaluate predicate
        active = (l_shipdate[i] >= lower_shipdate) && (l_shipdate[i] < upper_shipdate);

        const auto summand = l_extendedprice[i] * (100 - l_discount[i]);
        const auto partkey = l_partkey[i];

        // TODO materialize

        // determine all threads with matching tuples
        uint32_t mask = __ballot_sync(FULL_MASK, active);
        const auto count = __popc(mask);
        if (count < 1) continue;

        // flush buffer
        if (count + buffer_count > per_warp_buffer_size) {
            // reserve space in global materialization buffer
            uint32_t base;
            if (my_lane == 0) {
                base = atomicAdd(args.materialized_size, buffer_count);
            }
            base = __shfl_sync(FULL_MASK, base, 0);

            // materialize into global buffer
            for (unsigned j = 0; j < tuples_per_thread; ++j) {
                unsigned thread_offset = j * (tuples_per_thread * 32) + my_lane;
                const auto& src_tuple = tuple_buffer[thread_offset];
                auto& dst_tuple = args.materialized[base + thread_offset];
                dst_tuple.summand = src_tuple.summand;
                dst_tuple.l_partkey = src_tuple.summand;
            }

            // reset buffer count
            buffer_count = 0;
        }

        // update warp buffer index
        const uint32_t right_mask = __funnelshift_l(FULL_MASK, 0, my_lane);
        const unsigned thread_offset = __popc(mask & right_mask);
        uint32_t base = warp_id * per_warp_buffer_size;
        if (my_lane == 0) {
            base += atomicAdd(buffer_count, count);
        }
        base = __shfl_sync(FULL_MASK, base, 0);

        if (active) {
            auto& tuple = tuple_buffer[base + thread_offset];
            tuple.summand = summand;
            tuple.l_partkey = l_partkey;
        }
    }

    // TODO compute prefix sum
}

template<class IndexStructureType>
//__global__ void partitioned_ij_lookup(const lineitem_table_plain_t* __restrict__ lineitem, unsigned lineitem_size, const IndexStructureType index_structure, JoinEntry* __restrict__ join_entries) {
__global__ void partitioned_ij_lookup(partitioned_index_join_args args) {
    const char* prefix = "PROMO";

    int64_t numerator = 0;
    int64_t denominator = 0;

    const auto* __restrict__ p_type = args.part.p_type;

    const auto* __restrict__ l_partkey = args.materialized.l_partkey; // TODO
    const auto* __restrict__ summand = args.materialized.l_partkey; // TODO


    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < args.materialized_size + 31; i += stride) {
        bool active = i < args.lineitem_size;

        payload_t part_tid = args.index_structure.cooperative_lookup(l_partkey[i]);

        if (part_tid != invalid_tid) {/*
            const auto extendedprice = lineitem->l_extendedprice[i];
            const auto discount = lineitem->l_discount[i];
            const auto summand = extendedprice * (100 - discount);*/
            denominator += summand[i];

            const char* type = reinterpret_cast<const char*>(&p_type[part_tid]); // FIXME relies on undefined behavior
            if (device_strcmp(type, prefix, 5) == 0) {
                sum1 += summand;
            }
        } else 
    }

    // reduce both sums
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        numerator += __shfl_down_sync(FULL_MASK, numerator, offset);
        denominator += __shfl_down_sync(FULL_MASK, denominator, offset);
    }
    if (lane_id() == 0) {
        atomicAdd((unsigned long long int*)args.global_numerator, (unsigned long long int)numerator);
        atomicAdd((unsigned long long int*)args.global_denominator, (unsigned long long int)denominator);
    }
}

template __global__ void partitioned_ij_lookup<btree_type>(partitioned_index_join_args args);
template __global__ void partitioned_ij_lookup<harmonia_type>(partitioned_index_join_args args);
template __global__ void partitioned_ij_lookup<lower_bound_type>(partitioned_index_join_args args);
template __global__ void partitioned_ij_lookup<radix_spline_type>(partitioned_index_join_args args);
template __global__ void partitioned_ij_lookup<no_op_type>(partitioned_index_join_args args);
