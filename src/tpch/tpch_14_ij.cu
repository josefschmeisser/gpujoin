#include "config.hpp"
#include "tpch_14_ij.cuh"
#include "tpch_14_common.cuh"

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
#include <unordered_map>

#include <cub/block/block_radix_sort.cuh>

#include "common.hpp"
#include "cuda_utils.cuh"
#include "cuda_allocator.hpp"
#include "numa_allocator.hpp"
#include "mmap_allocator.hpp"
#include "indexes.cuh"
#include "device_array.hpp"

//#define MEASURE_CYCLES
//#define SKIP_SORT

using namespace cub;

template<class IndexStructureType>
__global__ void ij_plain_kernel(const ij_args args, const IndexStructureType index_structure) {
    const char* prefix = "PROMO";

    const auto* __restrict__ l_shipdate_column = args.lineitem->l_shipdate;
    const auto* __restrict__ l_extendedprice_column = args.lineitem->l_extendedprice;
    const auto* __restrict__ l_discount_column = args.lineitem->l_discount;
    const auto* __restrict__ l_partkey_column = args.lineitem->l_partkey;
    const auto* __restrict__ p_type_column = args.part->p_type;

    numeric_raw_t numerator = 0;
    numeric_raw_t denominator = 0;

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int loop_limit = (args.lineitem_size + warpSize - 1) & ~(warpSize - 1); // round to next multiple of warpSize
    for (int i = index; i < loop_limit; i += stride) {
        bool active = i < args.lineitem_size;

        // TODO check what this compiles to
        active &= l_shipdate_column[i] >= lower_shipdate && l_shipdate_column[i] >= upper_shipdate;

        const auto payload = index_structure.cooperative_lookup(active, l_partkey_column[i]);
        if (active && payload != invalid_tid) {
            const auto part_tid = reinterpret_cast<unsigned>(payload);

            const auto extendedprice = l_extendedprice_column[i];
            const auto discount = l_discount_column[i];
            const auto summand = extendedprice * (100 - discount);
            denominator += summand;

            const char* type = reinterpret_cast<const char*>(&p_type_column[part_tid]); // FIXME relies on undefined behavior
            if (device_strcmp(type, prefix, 5) == 0) {
                numerator += summand;
            }
        }
    }

    // reduce both sums
    auto* global_numerator = &args.state->global_numerator;
    auto* global_denominator = &args.state->global_denominator;

    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        numerator += __shfl_down_sync(FULL_MASK, numerator, offset);
        denominator += __shfl_down_sync(FULL_MASK, denominator, offset);
    }
    if (lane_id() == 0) {
        tmpl_atomic_add(global_numerator, numerator);
        tmpl_atomic_add(global_denominator, denominator);
    }
}

template __global__ void ij_plain_kernel<btree_type::device_index_t>(const ij_args args, const btree_type::device_index_t index_structure);
template __global__ void ij_plain_kernel<harmonia_type::device_index_t>(const ij_args args, const harmonia_type::device_index_t index_structure);
template __global__ void ij_plain_kernel<binary_search_type::device_index_t>(const ij_args args, const binary_search_type::device_index_t index_structure);
template __global__ void ij_plain_kernel<radix_spline_type::device_index_t>(const ij_args args, const radix_spline_type::device_index_t index_structure);
template __global__ void ij_plain_kernel<no_op_type::device_index_t>(const ij_args args, const no_op_type::device_index_t index_structure);

// Pipelined Blockwise Sorting index join kernel
template<
    unsigned BLOCK_THREADS,
    unsigned ITEMS_PER_THREAD,
    class    IndexStructureType >
__launch_bounds__ (BLOCK_THREADS)
__global__ void ij_pbws(const ij_args args, const IndexStructureType index_structure)
{
    assert(BLOCK_THREADS == blockDim.x);

#ifdef MEASURE_CYCLES
    const auto kernel_start = clock64();
#endif

    enum {
        ITEMS_PER_WARP = ITEMS_PER_THREAD * 32, // soft upper limit
        ITEMS_PER_BLOCK = BLOCK_THREADS*ITEMS_PER_THREAD,
        WARPS_PER_BLOCK = BLOCK_THREADS / 32,
        // the last summand ensures that each thread can write one more element during the last scan iteration
        BUFFER_SIZE = BLOCK_THREADS*(ITEMS_PER_THREAD + 1)
    };

    using BlockRadixSortT = cub::BlockRadixSort<indexed_t, BLOCK_THREADS, ITEMS_PER_THREAD, tid_t>;
    using key_array_t = indexed_t[ITEMS_PER_THREAD];
    using value_array_t = tid_t[ITEMS_PER_THREAD];

    const auto* __restrict__ l_shipdate_column = args.lineitem->l_shipdate;
    const auto* __restrict__ l_extendedprice_column = args.lineitem->l_extendedprice;
    const auto* __restrict__ l_discount_column = args.lineitem->l_discount;
    const auto* __restrict__ l_partkey_column = args.lineitem->l_partkey;
    const auto* __restrict__ p_type_column = args.part->p_type;

    auto* __restrict__ l_extendedprice_buffer = args.state->l_extendedprice_buffer + blockIdx.x*BUFFER_SIZE;
    auto* __restrict__ l_discount_buffer = args.state->l_discount_buffer + blockIdx.x*BUFFER_SIZE;

    __shared__ indexed_t l_partkey_buffer[BUFFER_SIZE];
    __shared__ tid_t lineitem_buffer_pos[BUFFER_SIZE];
    __shared__ int buffer_idx;

    __shared__ uint32_t fully_occupied_warps;
    __shared__ uint32_t exhausted_warps;

    __shared__ union {
        typename BlockRadixSortT::TempStorage temp_storage;
    } temp_union;

    uint32_t max_partkey = 0;

    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int max_reuse = warp_id*ITEMS_PER_WARP;
    const uint32_t right_mask = __funnelshift_l(FULL_MASK, 0, lane_id);

    const unsigned tile_size = min(args.lineitem_size, (args.lineitem_size + gridDim.x - 1) / gridDim.x);
    tid_t tid = blockIdx.x*tile_size; // first tid where the first thread of a block starts scanning
    const tid_t tid_limit = min(tid + tile_size, static_cast<tid_t>(args.lineitem_size)); // marks the end of each tile
    tid += threadIdx.x; // each thread starts at it's correponding offset

//if (lane_id == 0 && warp_id == 0) printf("lineitem_size: %u, gridDim.x: %u, tile_size: %u\n", lineitem_size, gridDim.x, tile_size);

    vector_to_raw_t<decltype(lineitem_table_t::l_shipdate)> l_shipdate;
    vector_to_raw_t<decltype(lineitem_table_t::l_partkey)> l_partkey;
    vector_to_raw_t<decltype(lineitem_table_t::l_extendedprice)> l_extendedprice;
    vector_to_raw_t<decltype(lineitem_table_t::l_discount)> l_discount;

    numeric_raw_t numerator = 0;
    numeric_raw_t denominator = 0;

    // initialize shared variables
    if (warp_id == 0 && lane_id == 0) {
        buffer_idx = 0;
        fully_occupied_warps = 0;
        exhausted_warps = 0;
    }
    __syncthreads(); // ensure that all shared variables are initialized

    uint32_t unexhausted_lanes = FULL_MASK; // lanes which can still fetch new tuples

    while (exhausted_warps < WARPS_PER_BLOCK || buffer_idx > 0) {
        // number of items stored in the buffer by this warp
        int warp_items = min(ITEMS_PER_WARP, max(0, buffer_idx - max_reuse));
//if (lane_id == 0) printf("warp: %d reuse: %u\n", warp_id, warp_items);

#ifdef MEASURE_CYCLES
        const auto t1 = clock64();
#endif
        while (unexhausted_lanes && warp_items < ITEMS_PER_WARP) {
            int active = tid < tid_limit;

            // fetch attributes
            if (active) {
                l_shipdate = l_shipdate_column[tid];
            }

            // filter predicate
            active = active && l_shipdate >= lower_shipdate && l_shipdate < upper_shipdate;

            // fetch remaining attributes
            if (active) {
                l_partkey = l_partkey_column[tid];
                l_extendedprice = l_extendedprice_column[tid];
                l_discount = l_discount_column[tid];
            }

            // negotiate buffer target positions among all threads in this warp
            const uint32_t active_mask = __ballot_sync(FULL_MASK, active);
            const auto item_cnt = __popc(active_mask);
            warp_items += item_cnt;
            uint32_t dest_idx = 0;
            if (lane_id == 0) {
                dest_idx = atomicAdd(&buffer_idx, item_cnt);
#ifndef NDEBUG
                atomicAdd(&args.state->lineitem_matches, item_cnt);
#endif
            }
            dest_idx = __shfl_sync(FULL_MASK, dest_idx, 0); // propagate the first buffer target index
            dest_idx += __popc(active_mask & right_mask); // add each participating thread's offset

            // matrialize attributes
            if (active) {
                lineitem_buffer_pos[dest_idx] = dest_idx;
                l_partkey_buffer[dest_idx] = l_partkey;
                l_discount_buffer[dest_idx] = l_discount;
                l_extendedprice_buffer[dest_idx] = l_extendedprice;
                max_partkey = (l_partkey > max_partkey) ? l_partkey : max_partkey;
            }

            unexhausted_lanes = __ballot_sync(FULL_MASK, tid < tid_limit);
            if (unexhausted_lanes == 0 && lane_id == 0) {
                atomicInc(&exhausted_warps, UINT_MAX);
            }

            tid += BLOCK_THREADS; // each tile is organized as a consecutive succession of elements belonging to the current thread block
        }

        if (lane_id == 0 && warp_items >= ITEMS_PER_WARP) {
            atomicInc(&fully_occupied_warps, UINT_MAX);
        }
#ifdef MEASURE_CYCLES
        __syncwarp();
        const auto scan_t2 = clock64();
        if (lane_id == 0) {
            atomicAdd(&scan_cycles, (unsigned long long)scan_t2 - t1);
        }
#endif

        __syncthreads(); // wait until all threads have gathered enough elements
#ifdef MEASURE_CYCLES
        __syncwarp();
        const auto sync_t2 = clock64();
        if (lane_id == 0) {
            atomicAdd(&sync_cycles, (unsigned long long)sync_t2 - scan_t2);
        }
#endif


#ifndef SKIP_SORT
        if (fully_occupied_warps == WARPS_PER_BLOCK) {
//if (warp_id == 0 && lane_id == 0) printf("=== sorting... ===\n");

            const unsigned first_offset = max(0, static_cast<int>(buffer_idx) - ITEMS_PER_BLOCK);

//if (warp_id == 0 && lane_id == 0) printf("=== first_offset: %u\n", first_offset);
            indexed_t* thread_keys_raw = reinterpret_cast<indexed_t*>(&l_partkey_buffer[threadIdx.x*ITEMS_PER_THREAD + first_offset]);
            tid_t* thread_values_raw = reinterpret_cast<tid_t*>(&lineitem_buffer_pos[threadIdx.x*ITEMS_PER_THREAD + first_offset]);
            key_array_t& thread_keys = reinterpret_cast<key_array_t&>(*thread_keys_raw);
            value_array_t& thread_values = reinterpret_cast<value_array_t&>(*thread_values_raw);

            BlockRadixSortT(temp_union.temp_storage).SortDescending(thread_keys, thread_values, 4, 22);
             __syncthreads();
        }
#endif
#ifdef MEASURE_CYCLES
        __syncwarp();
        const auto sort_t2 = clock64();
        if (lane_id == 0) {
            atomicAdd(&sort_cycles, (unsigned long long)sort_t2 - sync_t2);
        }
#endif

        // empty buffer
        for (unsigned i = 0u; i < ITEMS_PER_THREAD; ++i) {
            unsigned old;
            if (lane_id == 0) {
                // T atomic_sub_safe(T* address, T val)
                old = atomic_sub_safe(&buffer_idx, 32);
            }
            old = __shfl_sync(FULL_MASK, old, 0);
            const auto acquired_cnt = min(old, 32);
            const auto first_pos = old - acquired_cnt;
//if (lane_id == 0) printf("warp: %d iteration: %d - actual_count: %u\n", warp_id, i, actual_count);
//if (lane_id == 0) atomicAdd(&debug_cnt, acquired_cnt);

            if (acquired_cnt == 0u) break;

            bool active = lane_id < acquired_cnt;

            uint32_t assoc_pos = 0u;
            key_t l_partkey;
            if (active) {
                assoc_pos = lineitem_buffer_pos[first_pos + acquired_cnt - 1 - lane_id];
                l_partkey = l_partkey_buffer[first_pos + acquired_cnt - 1 - lane_id];
//printf("warp: %d lane: %d - tid: %u l_partkey: %u\n", warp_id, lane_id, assoc_pos, l_partkey);
            }

#ifdef MEASURE_CYCLES
            const auto lookup_t1 = clock64();
#endif
            tid_t tid_b = index_structure.cooperative_lookup(active, l_partkey);
            assert(!active || tid_b != invalid_tid);

#ifdef MEASURE_CYCLES
            __syncwarp();
            const auto lookup_t2 = clock64();
            if (lane_id == 0) {
                atomicAdd(&lookup_cycles, (unsigned long long)lookup_t2 - lookup_t1);
            }
#endif

            active = active && (tid_b != invalid_tid);

            if (active) {
                const auto summand = l_extendedprice_buffer[assoc_pos] * (100 - l_discount_buffer[assoc_pos]);
                denominator += summand;

                const char* type = reinterpret_cast<const char*>(&p_type_column[tid_b]); // FIXME relies on undefined behavior
                if (device_strcmp(type, "PROMO", 5) == 0) {
                    numerator += summand;
                }
            }

//printf("warp: %d lane: $d - element: %u\n", warp_id, lane_id, );
        }

#ifdef MEASURE_CYCLES
        __syncwarp();
        const auto join_t2 = clock64();
        if (lane_id == 0) {
            atomicAdd(&join_cycles, (unsigned long long)join_t2 - sort_t2);
        }
#endif

        // prepare next iteration
        if (lane_id == 0) {
            fully_occupied_warps = 0;
        }

        __syncthreads();
    }

    // finalize
    // reduce both sums
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        numerator += __shfl_down_sync(FULL_MASK, numerator, offset);
        denominator += __shfl_down_sync(FULL_MASK, denominator, offset);
    }
    if (lane_id == 0) {
        atomicAdd((unsigned long long int*)&args.state->global_numerator, (unsigned long long int)numerator);
        atomicAdd((unsigned long long int*)&args.state->global_denominator, (unsigned long long int)denominator);
    }

#ifdef MEASURE_CYCLES
    __syncwarp();
    const auto kernel_end = clock64();
    if (lane_id == 0) {
        atomicAdd(&total_cycles, (unsigned long long)kernel_end - kernel_start);
    }
#endif
}

template __global__ void ij_pbws<256, 10, btree_type::device_index_t>(const ij_args args, const btree_type::device_index_t index_structure);
template __global__ void ij_pbws<256, 10, harmonia_type::device_index_t>(const ij_args args, const harmonia_type::device_index_t index_structure);
template __global__ void ij_pbws<256, 10, binary_search_type::device_index_t>(const ij_args args, const binary_search_type::device_index_t index_structure);
template __global__ void ij_pbws<256, 10, radix_spline_type::device_index_t>(const ij_args args, const radix_spline_type::device_index_t index_structure);
template __global__ void ij_pbws<256, 10, no_op_type::device_index_t>(const ij_args args, const no_op_type::device_index_t index_structure);

// Non-Pipelined Plain Scan and Lookup kernel
template<class IndexStructureType>
__global__ void ij_lookup_kernel(const ij_args args, const IndexStructureType index_structure) {
    const auto* __restrict__ l_shipdate_column = args.lineitem->l_shipdate;
    const auto* __restrict__ l_partkey_column = args.lineitem->l_partkey;
    auto* __restrict__ join_entries = args.state->join_entries;

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int loop_limit = (args.lineitem_size + warpSize - 1) & ~(warpSize - 1); // round to next multiple of warpSize
    for (int i = index; i < loop_limit; i += stride) {
        bool active = i < args.lineitem_size;

        // TODO check what this compiles to
        active &= l_shipdate_column[i] >= lower_shipdate && l_shipdate_column[i] >= upper_shipdate;

        const auto payload = index_structure.cooperative_lookup(active, l_partkey_column[i]);

        int match = payload != invalid_tid;
        unsigned mask = __ballot_sync(FULL_MASK, match);
        unsigned my_lane = lane_id();
        unsigned right = __funnelshift_l(FULL_MASK, 0, my_lane);
        unsigned offset = __popc(mask & right);

        unsigned base = 0;
        int leader = __ffs(mask) - 1;
        if (my_lane == leader) {
            base = atomicAdd(&args.state->output_index, __popc(mask));
        }
        base = __shfl_sync(FULL_MASK, base, leader);

        if (match) {
            auto& join_entry = join_entries[base + offset];
            join_entry.lineitem_tid = i;
            join_entry.part_tid = payload;
        }
    }
}

template __global__ void ij_lookup_kernel<btree_type::device_index_t>(const ij_args args, const btree_type::device_index_t index_structure);
template __global__ void ij_lookup_kernel<harmonia_type::device_index_t>(const ij_args args, const harmonia_type::device_index_t index_structure);
template __global__ void ij_lookup_kernel<binary_search_type::device_index_t>(const ij_args args, const binary_search_type::device_index_t index_structure);
template __global__ void ij_lookup_kernel<radix_spline_type::device_index_t>(const ij_args args, const radix_spline_type::device_index_t index_structure);
template __global__ void ij_lookup_kernel<no_op_type::device_index_t>(const ij_args args, const no_op_type::device_index_t index_structure);

// Non-Pipelined Plain Join kernel
__global__ void ij_join_kernel(const ij_args args) {
    const char* prefix = "PROMO";

    const auto* __restrict__ l_extendedprice_column = args.lineitem->l_extendedprice;
    const auto* __restrict__ l_discount_column = args.lineitem->l_discount;
    const auto* __restrict__ p_type_column = args.part->p_type;
    const auto* __restrict__ join_entries = args.state->join_entries;

    numeric_raw_t numerator = 0;
    numeric_raw_t denominator = 0;

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const auto join_count = args.state->output_index;
    for (int i = index; i < join_count; i += stride) {
        const auto lineitem_tid = join_entries[i].lineitem_tid;
        const auto part_tid = join_entries[i].part_tid;

        const auto extendedprice = l_extendedprice_column[lineitem_tid];
        const auto discount = l_discount_column[lineitem_tid];
        const auto summand = extendedprice * (100 - discount);
        denominator += summand;

        const char* type = reinterpret_cast<const char*>(&p_type_column[part_tid]); // FIXME relies on undefined behavior
        if (device_strcmp(type, prefix, 5) == 0) {
            numerator += summand;
        }
    }

    // reduce both sums
    auto* global_numerator = &args.state->global_numerator;
    auto* global_denominator = &args.state->global_denominator;

    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        numerator += __shfl_down_sync(FULL_MASK, numerator, offset);
        denominator += __shfl_down_sync(FULL_MASK, denominator, offset);
    }
    //__reduce_add_sync() requires compute capability 8
    if (lane_id() == 0) {
        tmpl_atomic_add(global_numerator, numerator);
        tmpl_atomic_add(global_denominator, denominator);
    }
}

// ----------------------------------------------------------------------------


// Non-Pipelined Lane Refill Scan and Lookup kernel
// ij_lookup_kernel_4
template<
    int   BLOCK_THREADS,
    class IndexStructureType >
__global__ void ij_np_lane_refill_scan_lookup(const ij_args args, const IndexStructureType index_structure) {
    const auto* __restrict__ l_shipdate_column = args.lineitem->l_shipdate;
    const auto* __restrict__ l_partkey_column = args.lineitem->l_partkey;
    auto* __restrict__ join_entries = args.state->join_entries;

    const unsigned my_warp = threadIdx.x / 32;
    const unsigned my_lane = lane_id();
    const uint32_t right_mask = __funnelshift_l(FULL_MASK, 0, my_lane);

    __shared__ uint32_t l_partkey_buffer[BLOCK_THREADS];
    __shared__ uint32_t lineitem_buffer_pos[BLOCK_THREADS];

    // attributes
    vector_to_raw_t<decltype(lineitem_table_t::l_shipdate)> l_shipdate;
    vector_to_raw_t<decltype(lineitem_table_t::l_partkey)> l_partkey;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    tid_t lineitem_tid;
    unsigned buffer_cnt = 0; // number of buffered items in this warp
    const unsigned buffer_start = 32*my_warp;

    uint32_t unfinished_lanes = __ballot_sync(FULL_MASK, index < args.lineitem_size);
    while (unfinished_lanes || buffer_cnt > 0) {
        bool active = index < args.lineitem_size;
        if (active) {
            l_shipdate = l_shipdate_column[index];
            l_partkey = l_partkey_column[index];
            lineitem_tid = index;
        }

        if (active) {
            active = (l_shipdate >= lower_shipdate && l_shipdate < upper_shipdate);
        }

        const auto active_mask = __ballot_sync(FULL_MASK, active);
        auto active_cnt = __popc(active_mask);

        const unsigned threshold_cnt = (unfinished_lanes == 0) ? 0 : 25;
        while (buffer_cnt + active_cnt > threshold_cnt) {

            if (active_cnt < 25 && buffer_cnt > 0) {
                // refill
                const unsigned offset = __popc((~active_mask) & right_mask);

                const unsigned refill_cnt = min(buffer_cnt, 32 - active_cnt);

                if (!active && offset < buffer_cnt) {
                    const unsigned buffer_idx = buffer_start + buffer_cnt - offset - 1;
                    l_partkey = l_partkey_buffer[buffer_idx];
                    lineitem_tid = lineitem_buffer_pos[buffer_idx];
                    active = true;
                }

                buffer_cnt -= refill_cnt;
            }

            // next operator
            tid_t payload = index_structure.cooperative_lookup(active, l_partkey);

            const int match = payload != invalid_tid;
            const uint32_t mask = __ballot_sync(FULL_MASK, active && match);
            const unsigned offset = __popc(mask & right_mask);

            unsigned base = 0;
            if (my_lane == 0 && mask) {
                base = atomicAdd(&args.state->output_index, __popc(mask));
            }
            base = __shfl_sync(FULL_MASK, base, 0);

            if (active && match) {
                auto& join_entry = join_entries[base + offset];
                join_entry.lineitem_tid = lineitem_tid;
                join_entry.part_tid = payload;
            }

            active = false;
            active_cnt = 0;
        }

        if (active_cnt > 0) {
            // fill buffer
            const unsigned offset = __popc(active_mask & right_mask);
            if (active) {
                const unsigned buffer_idx = buffer_start + buffer_cnt + offset;
                l_partkey_buffer[buffer_idx] = l_partkey;
                lineitem_buffer_pos[buffer_idx] = lineitem_tid;
            }
            __syncwarp();

            buffer_cnt += active_cnt;
        }

        index += stride;
        unfinished_lanes = __ballot_sync(FULL_MASK, index < args.lineitem_size);
    }
}

template __global__ void ij_np_lane_refill_scan_lookup<256, btree_type::device_index_t>(const ij_args args, const btree_type::device_index_t index_structure);
template __global__ void ij_np_lane_refill_scan_lookup<256, harmonia_type::device_index_t>(const ij_args args, const harmonia_type::device_index_t index_structure);
template __global__ void ij_np_lane_refill_scan_lookup<256, binary_search_type::device_index_t>(const ij_args args, const binary_search_type::device_index_t index_structure);
template __global__ void ij_np_lane_refill_scan_lookup<256, radix_spline_type::device_index_t>(const ij_args args, const radix_spline_type::device_index_t index_structure);
template __global__ void ij_np_lane_refill_scan_lookup<256, no_op_type::device_index_t>(const ij_args args, const no_op_type::device_index_t index_structure);







#if 0

template<
    int   BLOCK_THREADS,
    int   ITEMS_PER_THREAD,
    class IndexStructureType >
__launch_bounds__ (BLOCK_THREADS)
__global__ void ij_lookup_kernel_2(
    const lineitem_table_plain_t* __restrict__ lineitem,
    const unsigned lineitem_size,
    IndexStructureType index_structure,
    join_entry* __restrict__ join_entries)
{
    enum {
        MAX_ITEMS_PER_WARP = ITEMS_PER_THREAD * 32,
        WARPS_PER_BLOCK = BLOCK_THREADS / 32,
        // the last summand ensures that each thread can write one more element during the last scan iteration
        BUFFER_SIZE = ITEMS_PER_THREAD*BLOCK_THREADS + BLOCK_THREADS,
        BUFFER_SOFT_LIMIT = ITEMS_PER_THREAD*BLOCK_THREADS
    };
    typedef BlockRadixSort<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;

    __shared__ uint32_t l_partkey_buffer[BUFFER_SIZE];
    __shared__ uint32_t lineitem_buffer_pos[BUFFER_SIZE];
    __shared__ uint32_t buffer_idx;

    __shared__ uint32_t fully_occupied_warps;
    __shared__ uint32_t exhausted_warps;

    __shared__ typename BlockRadixSortT::TempStorage temp_storage;

    /*
    union {
        struct {
            uint32_t l_partkey;
            uint32_t lineitem_tid;
        } join_pair;
        uint64_t raw;
    } join_pairs[ITEMS_PER_THREAD];
*/
    union {
        uint64_t join_pairs_raw[ITEMS_PER_THREAD];
        struct {
            uint32_t l_partkey;
            uint32_t lineitem_tid;
        } join_pairs[ITEMS_PER_THREAD];
    };

    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    const unsigned tile_size = min(lineitem_size, (lineitem_size + gridDim.x - 1) / gridDim.x);
    unsigned tid = blockIdx.x*tile_size; // first tid where the first thread of a block starts scanning
    const unsigned tid_limit = min(tid + tile_size, lineitem_size); // marks the end of each tile
    tid += threadIdx.x; // each thread starts at it's correponding offset

    // initialize shared variables
    if (warp_id == 0 && lane_id == 0) {
        buffer_idx = 0;
        fully_occupied_warps = 0;
        exhausted_warps = 0;
    }
    __syncthreads(); // ensure that all shared variables are initialized

    uint32_t unexhausted_lanes = FULL_MASK; // lanes which can still fetch new tuples

    while (exhausted_warps < WARPS_PER_BLOCK || buffer_idx > 0) {
        uint16_t local_idx = 0; // current size of the thread local array
        uint32_t underfull_lanes = FULL_MASK; // lanes that have less than ITEMS_PER_THREAD items in their registers (has to be reset after each iteration)

        while (unexhausted_lanes && underfull_lanes && buffer_idx < BUFFER_SOFT_LIMIT) {
            int active = tid < tid_limit;

            // TODO vectorize loads

            // filter predicate
            if (active) {
                active = lineitem->l_shipdate[tid] >= lower_shipdate && lineitem->l_shipdate[tid] < upper_shipdate;
            }

            // fetch attributes
            uint32_t l_partkey;
            if (active) {
                l_partkey = lineitem->l_partkey[tid];
            }

            // negotiate buffer target positions among all threads in this warp
            const uint32_t overflow_lanes = __ballot_sync(FULL_MASK, active && local_idx >= ITEMS_PER_THREAD);
            const uint32_t right = __funnelshift_l(FULL_MASK, 0, lane_id);
            uint32_t dest_idx = 0;
            if (overflow_lanes != 0 && lane_id == 0) {
                dest_idx = atomicAdd(&buffer_idx, __popc(overflow_lanes));
            }
            const uint32_t lane_offset = __popc(overflow_lanes & right);
            dest_idx = lane_offset + __shfl_sync(FULL_MASK, dest_idx, 0);

            // matrialize attributes
            if (active && local_idx >= ITEMS_PER_THREAD) {
                // buffer items
                lineitem_buffer_pos[dest_idx] = tid;
                l_partkey_buffer[dest_idx] = l_partkey;
            } else if (active) {
                // store items in registers
                auto& p = join_pairs[local_idx++];
                p.lineitem_tid = tid;
                p.l_partkey = l_partkey;
            }

            underfull_lanes = __ballot_sync(FULL_MASK, local_idx < ITEMS_PER_THREAD);
            unexhausted_lanes = __ballot_sync(FULL_MASK, tid < tid_limit);

            if (unexhausted_lanes == 0 && lane_id == 0) {
                atomicInc(&exhausted_warps, UINT_MAX);
            }

            tid += BLOCK_THREADS; // each tile is organized as a consecutive succession of its corresponding block
        }

        __syncthreads(); // wait until all threads have gathered enough elements

        // determine the number of items required to fully populate this lane
        const unsigned required = ITEMS_PER_THREAD - local_idx;
        int refill_cnt = 0;
        unsigned ideal_refill_cnt = required;

        // compute the number of required items across all lanes
        if (underfull_lanes) {
            #pragma unroll
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                ideal_refill_cnt += __shfl_down_sync(FULL_MASK, ideal_refill_cnt, offset);
            }
        }

        // distribute buffered items among the threads in this warp
        if (ideal_refill_cnt > 0) {
            uint32_t refill_idx_start;
            if (lane_id == 0) {
                const auto old = atomic_sub_safe(&buffer_idx, ideal_refill_cnt);
                refill_cnt = (old > ideal_refill_cnt) ? ideal_refill_cnt : old;
                refill_idx_start = old - refill_cnt;
            }

            refill_cnt = __shfl_sync(FULL_MASK, refill_cnt, 0);
            refill_idx_start = __shfl_sync(FULL_MASK, refill_idx_start, 0);

            int prefix_sum = required;
            // calculate the inclusive prefix sum among all threads in this warp
            #pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                auto value = __shfl_up_sync(FULL_MASK, prefix_sum, offset);
                prefix_sum += (lane_id >= offset) ? value : 0;
            }
            // calculate the exclusive prefix sum
            prefix_sum -= required;

            // refill registers with buffered elements
            const auto limit = min(prefix_sum + required, refill_cnt);
            for (; prefix_sum < limit; ++prefix_sum) {
                auto& p = join_pairs[local_idx++];
                p.lineitem_tid = lineitem_buffer_pos[refill_idx_start + prefix_sum];
                p.l_partkey = l_partkey_buffer[refill_idx_start + prefix_sum];
            }

            ideal_refill_cnt -= refill_cnt;
        }

        if (ideal_refill_cnt == 0 && lane_id == 0) {
            atomicInc(&fully_occupied_warps, UINT_MAX);
        }

        __syncthreads(); // wait until all threads have tried to fill their registers

        if (fully_occupied_warps == WARPS_PER_BLOCK) {/*
            if (warp_id == 0 && lane_id == 0) printf("=== sorting... ===\n");
            assert(join_pairs[0].l_partkey == (join_pairs_raw[0] & FULL_MASK));
*/

uint64_t* arr = nullptr;
typedef uint64_t items_t[ITEMS_PER_THREAD];
items_t& test = (items_t&)arr;

            BlockRadixSortT(temp_storage).SortBlockedToStriped(test, 8, 21); // TODO

        }

        unsigned output_base = 0;
        const auto count = MAX_ITEMS_PER_WARP - ideal_refill_cnt;
        if (lane_id == 0) {
            output_base = atomicAdd(&output_index, count);
        }
        output_base = __shfl_sync(FULL_MASK, output_base, 0);

        int lane_dst_idx_prefix_sum = local_idx;
        // calculate the inclusive prefix sum among all threads in this warp
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            auto value = __shfl_up_sync(FULL_MASK, lane_dst_idx_prefix_sum, offset);
            lane_dst_idx_prefix_sum += (lane_id >= offset) ? value : 0;
        }
        lane_dst_idx_prefix_sum -= local_idx;
// FIXME warp excution order is not deterministic
        uint32_t active_lanes = __ballot_sync(FULL_MASK, local_idx > 0);
        for (unsigned i = 0; active_lanes != 0; ++i) {
            bool active = i < local_idx;
            auto& p = join_pairs[i];
            const auto tid = index_structure.cooperative_lookup(active, p.l_partkey);

            if (active) {
                assert(tid != invalid_tid);
                auto& join_entry = join_entries[output_base + lane_dst_idx_prefix_sum++];
                join_entry.lineitem_tid = p.lineitem_tid;
                join_entry.part_tid = tid;
            }
            active_lanes = __ballot_sync(FULL_MASK, active);
        }

        // reset state
        __syncthreads(); // wait until each wrap is done
        if (lane_id == 0) {
            fully_occupied_warps = 0;
        }
    }
}

#if 0
template<
    unsigned BLOCK_THREADS,
    unsigned ITEMS_PER_THREAD,
    class    IndexStructureType >
__launch_bounds__ (BLOCK_THREADS)
__global__ void ij_lookup_kernel_3(
    const lineitem_table_plain_t* __restrict__ lineitem,
    const unsigned lineitem_size,
    const IndexStructureType index_structure,
    join_entry* __restrict__ join_entries)
{
    enum { ITEMS_PER_ITERATION = BLOCK_THREADS*ITEMS_PER_THREAD };

    // shared memory entry type
    union join_pair_t {
        struct { // TODO high low
            uint32_t l_partkey;
            uint32_t lineitem_tid;
        };
        uint64_t raw;
    };

    typedef cub::BlockRadixSort<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;

    __shared__ union TempStorage {
        // Allocate shared memory for BlockLoad
//        typename BlockLoad::TempStorage load;

        typename BlockRadixSortT::TempStorage sort;
    } temp_storage;

//    __shared__ uint64_t buffer[ITEMS_PER_ITERATION];
    __shared__ join_pair_t buffer[ITEMS_PER_ITERATION];
    __shared__ uint32_t buffer_pos;

    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    const unsigned tile_size = min(lineitem_size, (lineitem_size + gridDim.x - 1) / gridDim.x);
    unsigned tid_begin = blockIdx.x * tile_size; // first tid where scanning starts at each new iteration
    const unsigned tid_limit = ; // TODO
//if (lane_id == 0) printf("warp: %d tile_size: %d\n", warp_id, tile_size);

    const unsigned iteration_count = (tile_size + ITEMS_PER_ITERATION - 1) / ITEMS_PER_ITERATION;

    using key_value_array_t = uint64_t[ITEMS_PER_THREAD];
    uint64_t* thread_data_raw = &buffer[threadIdx.x*ITEMS_PER_THREAD];
    key_value_array_t& thread_data = reinterpret_cast<key_value_array_t&>(*thread_data_raw);

    for (unsigned i = 0; i < iteration_count; ++i) {
        // reset shared memory variables
        if (lane_id == 0) {
            buffer_pos = 0;
        }

//if (lane_id == 0) printf("warp: %d iteration: %d first tid: %d\n", warp_id, i, tid);

        unsigned valid_items = min(ITEMS_PER_ITERATION, lineitem_size - tid_begin);
//if (lane_id == 0) printf("warp: %d valid_items: %d\n", warp_id, valid_items);

        #pragma unroll
        for (unsigned j = 0; j < ITEMS_PER_THREAD; ++j) {
            // sort_buffer::produce
            tid_t payload = invalid_tid;
            unsigned lineitem_tid = tid_begin + threadIdx.x*ITEMS_PER_THREAD + j;
            if (lineitem_tid < lineitem_size &&
                lineitem->l_shipdate[lineitem_tid] >= lower_shipdate &&
                lineitem->l_shipdate[lineitem_tid] < upper_shipdate)
            {
//                payload = index_structure(lineitem->l_partkey[i]);
                auto& join_pair = buffer[threadIdx.x*ITEMS_PER_THREAD + j];
                join_pair.lineitem_tid = lineitem_tid;
                join_pair.l_partkey = lineitem->l_partkey[lineitem_tid];
                assert(join_pair.raw & 0xffffffff == lineitem_tid); // TODO
            }
        }

        __syncthreads();

#if 1
        // we only perform the sort step when the buffer is completely filled
        if (valid_items == ITEMS_PER_ITERATION) {
//if (lane_id == 0) printf("warp: %d iteration: %d - sorting... ===\n", warp_id, i);
            BlockRadixSortT(temp_storage.sort).Sort(thread_data, 4, 24); // TODO
             __syncthreads();
        }
#endif

        // empty buffer
        unsigned old;
        do {
            if (lane_id == 0) {
                old = atomic_add_sat(&buffer_pos, 32u, valid_items);
            }
            old = __shfl_sync(FULL_MASK, old, 0);
            unsigned actual_count = min(valid_items - old, 32);
//if (lane_id == 0) printf("warp: %d iteration: %d - actual_count: %u\n", warp_id, i, actual_count);

            if (actual_count == 0) break;

            bool active = lane_id < actual_count;

            uint32_t assoc_tid = 0;
            key_t element = 0xffffffff;
            if (active) {
                assoc_tid = buffer[old + lane_id] >> 32;
                element = buffer[old + lane_id] & 0xffffffff;
//printf("warp: %d lane: %d - tid: %u element: %u\n", warp_id, lane_id, assoc_tid, element);
            }

            const auto tid_b = index_structure.cooperative_lookup(active, element);
            if (active) {
//printf("warp: %d lane: %d - tid_b: %u\n", warp_id, lane_id, tid_b);
//                tids[assoc_tid] = tid_b;
                // TODO
            }

//printf("warp: %d lane: $d - element: %u\n", warp_id, lane_id, );

        } while (true);//actual_count == 32);


        tid_begin += valid_items;
    }
}
#endif







__managed__ unsigned debug_cnt = 0;

template<
    unsigned BLOCK_THREADS,
    unsigned ITEMS_PER_THREAD,
    class    IndexStructureType >
__launch_bounds__ (BLOCK_THREADS)
__global__ void ij_lookup_kernel_3(
    const lineitem_table_plain_t* __restrict__ lineitem,
    const unsigned lineitem_size,
    const IndexStructureType index_structure,
    join_entry* __restrict__ join_entries)
{
    enum {
        ITEMS_PER_WARP = ITEMS_PER_THREAD * 32, // soft upper limit
        ITEMS_PER_BLOCK = BLOCK_THREADS*ITEMS_PER_THREAD,
        WARPS_PER_BLOCK = BLOCK_THREADS / 32,
        // the last summand ensures that each thread can write one more element during the last scan iteration
//        BUFFER_SIZE = ITEMS_PER_THREAD*BLOCK_THREADS + BLOCK_THREADS,
        BUFFER_SIZE = BLOCK_THREADS*(ITEMS_PER_THREAD + 1)
//        BUFFER_SOFT_LIMIT = ITEMS_PER_THREAD*BLOCK_THREADS
    };

    // shared memory entry type
    union join_pair_t {
        struct { // TODO high low
            uint32_t l_partkey;
            uint32_t lineitem_tid;
        };
        uint64_t raw;
    };

    using BlockRadixSortT = cub::BlockRadixSort<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>;
    using key_value_array_t = uint64_t[ITEMS_PER_THREAD];

    __shared__ join_pair_t buffer[BUFFER_SIZE];
    __shared__ int buffer_idx;

    __shared__ uint32_t fully_occupied_warps;
    __shared__ uint32_t exhausted_warps;

    __shared__ union {
   //     uint16_t histogram[32]; // counts the respective msb
        typename BlockRadixSortT::TempStorage temp_storage;
    } temp_union;


    const float percentile = 0.9; // TODO
    const float r = 0.05;
    float moving_avg = -1.;
    float moving_seq_avg = -1.;
    float moving_percentile = -1.;

uint32_t max_partkey = 0;

    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int max_reuse = warp_id*ITEMS_PER_WARP;
    const uint32_t right_mask = __funnelshift_l(FULL_MASK, 0, lane_id);

    const unsigned tile_size = min(lineitem_size, (lineitem_size + gridDim.x - 1) / gridDim.x);
    unsigned tid = blockIdx.x*tile_size; // first tid where the first thread of a block starts scanning
    const unsigned tid_limit = min(tid + tile_size, lineitem_size); // marks the end of each tile
    tid += threadIdx.x; // each thread starts at it's correponding offset

//if (lane_id == 0 && warp_id == 0) printf("lineitem_size: %u, gridDim.x: %u, tile_size: %u\n", lineitem_size, gridDim.x, tile_size);

    // initialize shared variables
    if (warp_id == 0 && lane_id == 0) {
        buffer_idx = 0;
        fully_occupied_warps = 0;
        exhausted_warps = 0;
    }
    __syncthreads(); // ensure that all shared variables are initialized

    uint32_t unexhausted_lanes = FULL_MASK; // lanes which can still fetch new tuples

    while (exhausted_warps < WARPS_PER_BLOCK || buffer_idx > 0) {
        // number of items stored in the buffer by this warp
        int warp_items = min(ITEMS_PER_WARP, max(0, buffer_idx - max_reuse));

        while (unexhausted_lanes && warp_items < ITEMS_PER_WARP) {
            int active = tid < tid_limit;

            // TODO vectorize loads

            // filter predicate
            if (active) {
                active = lineitem->l_shipdate[tid] >= lower_shipdate && lineitem->l_shipdate[tid] < upper_shipdate;
            }

            // fetch attributes
            uint32_t l_partkey;
            if (active) {
                l_partkey = lineitem->l_partkey[tid];
            }

            // negotiate buffer target positions among all threads in this warp
            const uint32_t active_mask = __ballot_sync(FULL_MASK, active);
            const auto item_cnt = __popc(active_mask);
            warp_items += item_cnt;
            uint32_t dest_idx = 0;
            if (lane_id == 0) {
                dest_idx = atomicAdd(&buffer_idx, item_cnt);
 //atomicAdd(&debug_cnt, item_cnt);
            }
            dest_idx = __shfl_sync(FULL_MASK, dest_idx, 0); // propagate the first buffer target index
            dest_idx += __popc(active_mask & right_mask); // add each participating thread's offset

            // matrialize attributes
            if (active) {
                auto& join_pair = buffer[dest_idx];
                join_pair.l_partkey = l_partkey;
                join_pair.lineitem_tid = tid;
//                printf("tid: %u\n", tid);
                /*
                printf("raw: %p rtid: %u rkey: %u tid1: %p key1: %p tid2: %u key2: %u\n", tid, l_partkey, (void*)join_pair.raw, (void*)(join_pair.raw & 0xffffffff),
                        (void*)(join_pair.raw >> 32), (unsigned)(join_pair.raw & 0xffffffff), (unsigned)(join_pair.raw >> 32));
                        */
//                printf("raw: %p tid: %u key: %u\n", (void*)join_pair.raw, (unsigned)(join_pair.raw & 0xffffffff), (unsigned)(join_pair.raw >> 32));
//                __threadfence();
/*
                assert((join_pair.raw & 0xffffffff) == l_partkey); // TODO
                assert((join_pair.raw >> 32) == tid); // TODO
*/

#if 0
                // update moving percentile
                if (moving_percentile < 0.) {
                    // initialize
                    moving_avg = l_partkey;
                    moving_seq_avg = l_partkey*l_partkey;
                    moving_percentile = l_partkey;
                } else {
                    moving_avg = r*l_partkey + (1. - r)*moving_avg;
                    auto current_var = moving_avg - l_partkey;
                    current_var *= current_var;
                    moving_seq_avg = r*current_var + (1. - r)*moving_seq_avg;

                    if (moving_percentile > l_partkey) {
                        moving_percentile -= sqrtf(moving_seq_avg)*r/percentile;
                    } else if (moving_percentile < l_partkey) {
                        moving_percentile += sqrtf(moving_seq_avg)*r/(1. - percentile);
                    }
                }
#endif
                max_partkey = (l_partkey > max_partkey) ? l_partkey : max_partkey;
            }

/*
// compute moving percentile
unsigned g = __popc(__ballot_sync(FULL_MASK, active && l_partkey > moving_percentile));
unsigned l = __popc(__ballot_sync(FULL_MASK, active && l_partkey < moving_percentile));

// end
*/

            unexhausted_lanes = __ballot_sync(FULL_MASK, tid < tid_limit);
            if (unexhausted_lanes == 0 && lane_id == 0) {
                atomicInc(&exhausted_warps, UINT_MAX);
            }

            tid += BLOCK_THREADS; // each tile is organized as a consecutive succession of elements belonging to the current thread block
        }

        if (lane_id == 0 && warp_items >= ITEMS_PER_WARP) {
            atomicInc(&fully_occupied_warps, UINT_MAX);
        }

        __syncthreads(); // wait until all threads have gathered enough elements

//if (lane_id == 0) printf("moving_percentile: %f avg: %f max_partkey: %u diff %f\n", moving_percentile, moving_avg, max_partkey, static_cast<float>(max_partkey)-moving_percentile);

#if 1
        if (fully_occupied_warps == WARPS_PER_BLOCK) {
//if (warp_id == 0 && lane_id == 0) printf("=== sorting... ===\n");

            const unsigned first_offset = max(0, static_cast<int>(buffer_idx) - ITEMS_PER_BLOCK);
            uint64_t* thread_data_raw = reinterpret_cast<uint64_t*>(&buffer[threadIdx.x*ITEMS_PER_THREAD + first_offset]);
            key_value_array_t& thread_data = reinterpret_cast<key_value_array_t&>(*thread_data_raw);

            BlockRadixSortT(temp_union.temp_storage).SortDescending(thread_data, 4, 22); // TODO
             __syncthreads();
        }
#endif


#if 1
        // empty buffer
        for (unsigned i = 0u; i < ITEMS_PER_THREAD; ++i) {
            unsigned old;
            if (lane_id == 0) {
//                old = atomic_add_sat(&buffer_pos, 32u, valid_items);
                // T atomic_sub_safe(T* address, T val)
                old = atomic_sub_safe(&buffer_idx, 32);
            }
            old = __shfl_sync(FULL_MASK, old, 0);
            const auto acquired_cnt = min(old, 32);
            const auto first_pos = old - acquired_cnt;
//if (lane_id == 0) printf("warp: %d iteration: %d - actual_count: %u\n", warp_id, i, actual_count);
//if (lane_id == 0) atomicAdd(&debug_cnt, acquired_cnt);

            if (acquired_cnt == 0u) break;

            bool active = lane_id < acquired_cnt;

            uint32_t assoc_tid = 0u;
            key_t element;
            if (active) {
/*
                const auto my_pos = first_pos + 31u - lane_id;
                assoc_tid = buffer[my_pos] >> 32;
                element = buffer[my_pos] & 0xffffffff;
*/
//                const auto& join_pair = buffer[first_pos + 31u - lane_id];
                const auto& join_pair = buffer[first_pos + acquired_cnt - 1 - lane_id]; // TODO check
                assoc_tid = join_pair.lineitem_tid;
                element = join_pair.l_partkey;
//printf("warp: %d lane: %d - tid: %u element: %u\n", warp_id, lane_id, assoc_tid, element);
            }

            tid_t tid_b = index_structure.cooperative_lookup(active, element);

            active = active && (tid_b != invalid_tid);

            // negotiate output buffer target positions
            const uint32_t active_mask = __ballot_sync(FULL_MASK, active);
            const auto item_cnt = __popc(active_mask);
//assert(item_cnt == acquired_cnt);
            uint32_t dest_idx = 0;
            if (lane_id == 0) {
                dest_idx = atomicAdd(&output_index, item_cnt);
            }
            dest_idx = __shfl_sync(FULL_MASK, dest_idx, 0); // propagate the first buffer target index
            dest_idx += __popc(active_mask & right_mask); // add each's participating thread's offset

            // write entry into ouput buffer
            if (active) {
//printf("warp: %d lane: %d - tid_b: %u\n", warp_id, lane_id, tid_b);
                auto& join_entry = join_entries[dest_idx];
                join_entry.lineitem_tid = assoc_tid;
                join_entry.part_tid = tid_b;
            }

//printf("warp: %d lane: $d - element: %u\n", warp_id, lane_id, );

        }
#else
        // discard elements
        if (lane_id == 0) buffer_idx = 0;
#endif

        // prepare next iteration
        if (lane_id == 0) {
            fully_occupied_warps = 0;
        }

        // reset moving percentile
        moving_percentile = -1.;

        __syncthreads();
    }
}
#endif
