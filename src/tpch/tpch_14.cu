#include "common.hpp"

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

#include "cuda_utils.cuh"
#include "LinearProbingHashTable.cuh"
#include "cuda_allocator.hpp"
#include "numa_allocator.hpp"
#include "mmap_allocator.hpp"
#include "indexes.cuh"
#include "device_array.hpp"

#define MEASURE_CYCLES
#define SKIP_SORT

using namespace cub;

//using vector_copy_policy = vector_to_managed_array; // TODO remove

using indexed_t = std::remove_pointer_t<decltype(lineitem_table_plain_t::l_partkey)>;
using payload_t = uint32_t;

// host allocator
//template<class T> using host_allocator = mmap_allocator<T, huge_2mb, 1>;
template<class T> using host_allocator = std::allocator<T>;
//template<class T> using host_allocator = cuda_allocator<T, cuda_allocation_type::zero_copy>;
//template<class T> using host_allocator = mmap_allocator<T, huge_2mb, 0>;

// device allocators
template<class T> using device_index_allocator = cuda_allocator<T, cuda_allocation_type::zero_copy>;
template<class T> using device_table_allocator = cuda_allocator<T, cuda_allocation_type::zero_copy>;
//template<class T> using device_index_allocator = mmap_allocator<T, huge_2mb, 0>;
//template<class T> using device_table_allocator = mmap_allocator<T, huge_2mb, 0>;


static constexpr bool prefetch_index __attribute__((unused)) = false;
static constexpr bool sort_indexed_relation = true;
static constexpr int block_size = 128;
static int num_sms;

static const uint32_t lower_shipdate = 2449962; // 1995-09-01
static const uint32_t upper_shipdate = 2449992; // 1995-10-01
static const uint32_t invalid_tid __attribute__((unused)) = std::numeric_limits<uint32_t>::max();

__device__ unsigned int count = 0;
__managed__ int tupleCount;

using device_ht_t = LinearProbingHashTable<uint32_t, size_t>::DeviceHandle;

__global__ void hj_build_kernel(size_t n, const part_table_plain_t* part, device_ht_t ht) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < n; i += stride) {
        ht.insert(part->p_partkey[i], i);
    }
}

__device__ int my_strcmp(const char *str_a, const char *str_b, unsigned len){
    int match = 0;
    unsigned i = 0;
    unsigned done = 0;
    while ((i < len) && (match == 0) && !done) {
        if ((str_a[i] == 0) || (str_b[i] == 0)) {
            done = 1;
        } else if (str_a[i] != str_b[i]) {
            match = i+1;
            if (((int)str_a[i] - (int)str_b[i]) < 0) match = 0 - (i + 1);
        }
        i++;
    }
    return match;
}

__managed__ int64_t globalSum1 = 0;
__managed__ int64_t globalSum2 = 0;

__global__ void hj_probe_kernel(size_t n, const part_table_plain_t* __restrict__ part, const lineitem_table_plain_t* __restrict__ lineitem, device_ht_t ht) {
    const char* prefix = "PROMO";

    int64_t sum1 = 0;
    int64_t sum2 = 0;

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        if (lineitem->l_shipdate[i] < lower_shipdate ||
            lineitem->l_shipdate[i] >= upper_shipdate) {
            continue;
        }

        size_t part_tid;
        bool match = ht.lookup(lineitem->l_partkey[i], part_tid);
        // TODO refill
        if (match) {
            const auto extendedprice = lineitem->l_extendedprice[i];
            const auto discount = lineitem->l_discount[i];
            const auto summand = extendedprice * (100 - discount);
            sum2 += summand;

            const char* type = reinterpret_cast<const char*>(&part->p_type[part_tid]); // FIXME relies on undefined behavior
//            printf("type: %s\n", type);
            if (my_strcmp(type, prefix, 5) == 0) {
                sum1 += summand;
            }
        }
    }

    // reduce both sums
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum1 += __shfl_down_sync(FULL_MASK, sum1, offset);
        sum2 += __shfl_down_sync(FULL_MASK, sum2, offset);
    }
    if (lane_id() == 0) {
        atomicAdd((unsigned long long int*)&globalSum1, (unsigned long long int)sum1);
        atomicAdd((unsigned long long int*)&globalSum2, (unsigned long long int)sum2);
    }
}

template<class IndexStructureType>
__global__ void ij_full_kernel(const lineitem_table_plain_t* __restrict__ lineitem, const unsigned lineitem_size, const part_table_plain_t* __restrict__ part, IndexStructureType index_structure) {
    const char* prefix = "PROMO";

    int64_t sum1 = 0;
    int64_t sum2 = 0;

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = index; i < lineitem_size; i += stride) {
        if (lineitem->l_shipdate[i] < lower_shipdate ||
            lineitem->l_shipdate[i] >= upper_shipdate) {
            continue;
        }

        auto payload = index_structure.lookup(lineitem->l_partkey[i]);
        if (payload != invalid_tid) {
            const auto part_tid = reinterpret_cast<unsigned>(payload);

            const auto extendedprice = lineitem->l_extendedprice[i];
            const auto discount = lineitem->l_discount[i];
            const auto summand = extendedprice * (100 - discount);
            sum2 += summand;

            const char* type = reinterpret_cast<const char*>(&part->p_type[part_tid]); // FIXME relies on undefined behavior
//            printf("type: %s\n", type);
            if (my_strcmp(type, prefix, 5) == 0) {
                sum1 += summand;
            }
        }
    }

    // reduce both sums
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum1 += __shfl_down_sync(FULL_MASK, sum1, offset);
        sum2 += __shfl_down_sync(FULL_MASK, sum2, offset);
    }
    if (lane_id() == 0) {
        atomicAdd((unsigned long long int*)&globalSum1, (unsigned long long int)sum1);
        atomicAdd((unsigned long long int*)&globalSum2, (unsigned long long int)sum2);
    }
}

struct JoinEntry {
    unsigned lineitem_tid;
    unsigned part_tid;
};
__device__ unsigned output_index = 0;

/*
// source: https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
// increment the value at ptr by 1 and return the old value
__device__ int atomicAggInc(int *ptr) {
    int mask = __match_any_sync(__activemask(), (unsigned long long)ptr);
    int leader = __ffs(mask) - 1;    // select a leader
    int res;
    if(lane_id() == leader)                  // leader does the update
        res = atomicAdd(ptr, __popc(mask));
    res = __shfl_sync(mask, res, leader);    // get leader’s old value
    return res + __popc(mask & ((1 << lane_id()) - 1)); //compute old value
}*/

template<class IndexStructureType>
__global__ void ij_lookup_kernel(const lineitem_table_plain_t* __restrict__ lineitem, unsigned lineitem_size, const IndexStructureType index_structure, JoinEntry* __restrict__ join_entries) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = index; i < lineitem_size + 31; i += stride) {
        payload_t payload = invalid_tid;
        if (i < lineitem_size &&
            lineitem->l_shipdate[i] >= lower_shipdate &&
            lineitem->l_shipdate[i] < upper_shipdate) {
            payload = index_structure.lookup(lineitem->l_partkey[i]);
        }

        int match = payload != invalid_tid;
        unsigned mask = __ballot_sync(FULL_MASK, match);
        unsigned my_lane = lane_id();
        unsigned right = __funnelshift_l(0xffffffff, 0, my_lane);
//        printf("right %u\n", right);
        unsigned offset = __popc(mask & right);

        unsigned base = 0;
        int leader = __ffs(mask) - 1;
        if (my_lane == leader) {
            base = atomicAdd(&output_index, __popc(mask));
        }
        base = __shfl_sync(FULL_MASK, base, leader);

        if (match) {
//            printf("lane %u store to: %u\n", my_lane, base + offset);
            auto& join_entry = join_entries[base + offset];
            join_entry.lineitem_tid = i;
            join_entry.part_tid = payload;
        }
    }
}

__device__ __forceinline__ unsigned testfun() { return 0; }

template<
    int   BLOCK_THREADS,
    class IndexStructureType >
__global__ void ij_lookup_kernel_4(const lineitem_table_plain_t* __restrict__ lineitem, unsigned lineitem_size, const IndexStructureType index_structure, JoinEntry* __restrict__ join_entries) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const unsigned my_warp = threadIdx.x / 32;
    const unsigned my_lane = lane_id();
    const uint32_t right_mask = __funnelshift_l(0xffffffff, 0, my_lane);

    __shared__ uint32_t l_partkey_buffer[BLOCK_THREADS];
    __shared__ uint32_t lineitem_buffer_pos[BLOCK_THREADS];

    //__shared__ uint32_t buffer_idx;
    unsigned buffer_cnt = 0; // number of buffered items in this warp
    const unsigned buffer_start = 32*my_warp;

    // attributes
    uint32_t l_shipdate;
    uint32_t l_partkey;

    unsigned lineitem_tid;

    uint32_t unfinished_lanes = __ballot_sync(FULL_MASK, index < lineitem_size);
    while (unfinished_lanes || buffer_cnt > 0) {
        bool active = index < lineitem_size;
        if (active) {
            l_shipdate = lineitem->l_shipdate[index];
            l_partkey = lineitem->l_partkey[index];
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
            payload_t payload = index_structure.cooperative_lookup(active, l_partkey);

            const int match = payload != invalid_tid;
            const uint32_t mask = __ballot_sync(FULL_MASK, active && match);
            const unsigned offset = __popc(mask & right_mask);

            unsigned base = 0;
            if (my_lane == 0 && mask) {
                base = atomicAdd(&output_index, __popc(mask));
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
        unfinished_lanes = __ballot_sync(FULL_MASK, index < lineitem_size);
    }
}

template<
    int   BLOCK_THREADS,
    int   ITEMS_PER_THREAD,
    class IndexStructureType >
__launch_bounds__ (BLOCK_THREADS)
__global__ void ij_lookup_kernel_2(
    const lineitem_table_plain_t* __restrict__ lineitem,
    const unsigned lineitem_size,
    IndexStructureType index_structure,
    JoinEntry* __restrict__ join_entries)
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
    JoinEntry* __restrict__ join_entries)
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
            payload_t payload = invalid_tid;
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
    JoinEntry* __restrict__ join_entries)
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

            payload_t tid_b = index_structure.cooperative_lookup(active, element);

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




__managed__ unsigned long long lookup_cycles = 0;
__managed__ unsigned long long scan_cycles = 0;
__managed__ unsigned long long sync_cycles = 0;
__managed__ unsigned long long sort_cycles = 0;

template<
    unsigned BLOCK_THREADS,
    unsigned ITEMS_PER_THREAD,
    class    IndexStructureType >
__launch_bounds__ (BLOCK_THREADS)
__global__ void ij_full_kernel_2(
    const lineitem_table_plain_t* __restrict__ lineitem,
    const unsigned lineitem_size,
    const part_table_plain_t* __restrict__ part,
    const unsigned part_size,
    const IndexStructureType index_structure,
    int64_t* __restrict__ l_extendedprice_buffer,
    int64_t* __restrict__ l_discount_buffer
    )
{
    enum {
        ITEMS_PER_WARP = ITEMS_PER_THREAD * 32, // soft upper limit
        ITEMS_PER_BLOCK = BLOCK_THREADS*ITEMS_PER_THREAD,
        WARPS_PER_BLOCK = BLOCK_THREADS / 32,
        // the last summand ensures that each thread can write one more element during the last scan iteration
        BUFFER_SIZE = BLOCK_THREADS*(ITEMS_PER_THREAD + 1)
    };

    using BlockRadixSortT = cub::BlockRadixSort<uint32_t, BLOCK_THREADS, ITEMS_PER_THREAD, uint32_t>;
    using key_array_t = uint32_t[ITEMS_PER_THREAD];
    using value_array_t = uint32_t[ITEMS_PER_THREAD];

    l_extendedprice_buffer += blockIdx.x*BUFFER_SIZE;
    l_discount_buffer += blockIdx.x*BUFFER_SIZE;

    __shared__ uint32_t l_partkey_buffer[BUFFER_SIZE];
    __shared__ uint32_t lineitem_buffer_pos[BUFFER_SIZE];
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

    const unsigned tile_size = min(lineitem_size, (lineitem_size + gridDim.x - 1) / gridDim.x);
    unsigned tid = blockIdx.x*tile_size; // first tid where the first thread of a block starts scanning
    const unsigned tid_limit = min(tid + tile_size, lineitem_size); // marks the end of each tile
    tid += threadIdx.x; // each thread starts at it's correponding offset

//if (lane_id == 0 && warp_id == 0) printf("lineitem_size: %u, gridDim.x: %u, tile_size: %u\n", lineitem_size, gridDim.x, tile_size);

    uint32_t l_shipdate;
    uint32_t l_partkey;
    int64_t l_extendedprice;
    int64_t l_discount;

    int64_t sum1 = 0;
    int64_t sum2 = 0;

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
        const auto scan_t1 = clock64();
#endif
        while (unexhausted_lanes && warp_items < ITEMS_PER_WARP) {
            int active = tid < tid_limit;

            // fetch attributes
            if (active) {
                l_shipdate = lineitem->l_shipdate[tid];
            }

            // filter predicate
            active = active && l_shipdate >= lower_shipdate && l_shipdate < upper_shipdate;

            // fetch remaining attributes
            if (active) {
                l_partkey = lineitem->l_partkey[tid];
                l_extendedprice = lineitem->l_extendedprice[tid];
                l_discount = lineitem->l_discount[tid];
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
            atomicAdd(&scan_cycles, (unsigned long long)scan_t2 - scan_t1);
        }
#endif

#ifdef MEASURE_CYCLES
        const auto sync_t1 = clock64();
#endif
        __syncthreads(); // wait until all threads have gathered enough elements
#ifdef MEASURE_CYCLES
        __syncwarp();
        const auto sync_t2 = clock64();
        if (lane_id == 0) {
            atomicAdd(&sync_cycles, (unsigned long long)sync_t2 - sync_t1);
        }
#endif


#ifndef SKIP_SORT
#ifdef MEASURE_CYCLES
        const auto sort_t1 = clock64();
#endif
        if (fully_occupied_warps == WARPS_PER_BLOCK) {
//if (warp_id == 0 && lane_id == 0) printf("=== sorting... ===\n");

            const unsigned first_offset = max(0, static_cast<int>(buffer_idx) - ITEMS_PER_BLOCK);

//if (warp_id == 0 && lane_id == 0) printf("=== first_offset: %u\n", first_offset);
            uint32_t* thread_keys_raw = reinterpret_cast<uint32_t*>(&l_partkey_buffer[threadIdx.x*ITEMS_PER_THREAD + first_offset]);
            uint32_t* thread_values_raw = reinterpret_cast<uint32_t*>(&lineitem_buffer_pos[threadIdx.x*ITEMS_PER_THREAD + first_offset]);
            key_array_t& thread_keys = reinterpret_cast<key_array_t&>(*thread_keys_raw);
            value_array_t& thread_values = reinterpret_cast<value_array_t&>(*thread_values_raw);

            BlockRadixSortT(temp_union.temp_storage).SortDescending(thread_keys, thread_values, 4, 22);
             __syncthreads();
        }
#ifdef MEASURE_CYCLES
        __syncwarp();
        const auto sort_t2 = clock64();
        if (lane_id == 0) {
            atomicAdd(&sort_cycles, (unsigned long long)sort_t2 - sort_t1);
        }
#endif
#endif

#if 1
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
            payload_t tid_b = index_structure.cooperative_lookup(active, l_partkey);
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
                sum2 += summand;

                const char* type = reinterpret_cast<const char*>(&part->p_type[tid_b]); // FIXME relies on undefined behavior
                if (my_strcmp(type, "PROMO", 5) == 0) {
                    sum1 += summand;
                }
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

        __syncthreads();
    }

    // finalize
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







__global__ void ij_join_kernel(const lineitem_table_plain_t* __restrict__ lineitem, const part_table_plain_t* __restrict__ part, const JoinEntry* __restrict__ join_entries, size_t n) {
    int64_t sum1 = 0;
    int64_t sum2 = 0;
    const char* prefix = "PROMO";

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        const auto lineitem_tid = join_entries[i].lineitem_tid;
        const auto part_tid = join_entries[i].part_tid;

        const auto extendedprice = lineitem->l_extendedprice[lineitem_tid];
        const auto discount = lineitem->l_discount[lineitem_tid];
        const auto summand = extendedprice * (100 - discount);
        sum2 += summand;

        const char* type = reinterpret_cast<const char*>(&part->p_type[part_tid]); // FIXME relies on undefined behavior
//        printf("type: %s\n", type);
        if (my_strcmp(type, prefix, 5) == 0) {
            sum1 += summand;
        }
    }

    // reduce both sums
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum1 += __shfl_down_sync(FULL_MASK, sum1, offset);
        sum2 += __shfl_down_sync(FULL_MASK, sum2, offset);
    }
    //__reduce_add_sync() requires compute capability 8
    if (lane_id() == 0) {
        atomicAdd((unsigned long long int*)&globalSum1, (unsigned long long int)sum1);
        atomicAdd((unsigned long long int*)&globalSum2, (unsigned long long int)sum2);
    }
}

template<class IndexType>
struct helper {
    IndexType index_structure;

    unsigned lineitem_size;
    lineitem_table_plain_t* lineitem_device;
    std::unique_ptr<lineitem_table_plain_t> lineitem_device_ptrs;

    unsigned part_size;
    part_table_plain_t* part_device;
    std::unique_ptr<part_table_plain_t> part_device_ptrs;

    void load_database(const std::string& path) {
        Database db;
        load_tables(db, path);
        if (sort_indexed_relation) {
            printf("sorting part relation...\n");
            sort_relation(db.part);
        }
        lineitem_size = db.lineitem.l_orderkey.size();
        part_size = db.part.p_partkey.size();

        {
            using namespace std;
            const auto start = chrono::high_resolution_clock::now();
            device_table_allocator<int> a;
            //std::tie(lineitem_device, lineitem_device_ptrs) = copy_relation<vector_copy_policy>(db.lineitem);
            std::tie(lineitem_device, lineitem_device_ptrs) = migrate_relation(db.lineitem, a);
            //std::tie(part_device, part_device_ptrs) = copy_relation<vector_copy_policy>(db.part);
            std::tie(part_device, part_device_ptrs) = migrate_relation(db.part, a);
            const auto finish = chrono::high_resolution_clock::now();
            const auto d = chrono::duration_cast<chrono::milliseconds>(finish - start).count();
            std::cout << "transfer time: " << d << " ms\n";
        }

#ifndef USE_HJ
        index_structure.construct(db.part.p_partkey, part_device_ptrs->p_partkey);
#endif
    }

#ifdef USE_HJ
    void run_hj() {
        const auto kernelStart = std::chrono::high_resolution_clock::now();

        LinearProbingHashTable<uint32_t, size_t> ht(part_size);
        int num_blocks = (part_size + block_size - 1) / block_size;
        hj_build_kernel<<<num_blocks, block_size>>>(part_size, part_device, ht.deviceHandle);

        //num_blocks = 32*num_sms;
        num_blocks = (lineitem_size + block_size - 1) / block_size;
        hj_probe_kernel<<<num_blocks, block_size>>>(lineitem_size, part_device, lineitem_device, ht.deviceHandle);
        cudaDeviceSynchronize();

        const auto kernelStop = std::chrono::high_resolution_clock::now();
        const auto kernelTime = std::chrono::duration_cast<std::chrono::microseconds>(kernelStop - kernelStart).count()/1000.;
        std::cout << "kernel time: " << kernelTime << " ms\n";
    }
#endif

    void run_ij() {
        const auto kernelStart = std::chrono::high_resolution_clock::now();

        int num_blocks = (lineitem_size + block_size - 1) / block_size;
        ij_full_kernel<<<num_blocks, block_size>>>(lineitem_device, lineitem_size, part_device, index_structure.device_index);
        cudaDeviceSynchronize();

        const auto kernelStop = std::chrono::high_resolution_clock::now();
        const auto kernelTime = std::chrono::duration_cast<std::chrono::microseconds>(kernelStop - kernelStart).count()/1000.;
        std::cout << "kernel time: " << kernelTime << " ms\n";
    }

    void run_ij_buffer() {
        using namespace std;

        decltype(output_index) matches1 = 0;

        enum { BLOCK_THREADS = 256, ITEMS_PER_THREAD = 10 }; // TODO optimize

        JoinEntry* join_entries1;
        cudaMalloc(&join_entries1, sizeof(JoinEntry)*lineitem_size);

        int num_sms;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        int num_blocks = num_sms*4; // TODO


        int buffer_size = num_blocks*BLOCK_THREADS*(ITEMS_PER_THREAD + 1);
        int64_t* l_extendedprice_buffer;
        int64_t* l_discount_buffer;
        cudaMalloc(&l_extendedprice_buffer, sizeof(decltype(*l_extendedprice_buffer))*buffer_size);
        cudaMalloc(&l_discount_buffer, sizeof(decltype(*l_discount_buffer))*buffer_size);

        const auto kernelStart = std::chrono::high_resolution_clock::now();

        ij_full_kernel_2<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(lineitem_device, lineitem_size, part_device, part_size, index_structure.device_index, l_extendedprice_buffer, l_discount_buffer);
        cudaDeviceSynchronize();

        const auto kernelStop = std::chrono::high_resolution_clock::now();
        const auto kernelTime = std::chrono::duration_cast<std::chrono::microseconds>(kernelStop - kernelStart).count()/1000.;
        std::cout << "kernel time: " << kernelTime << " ms\n";
    }

    void run_two_phase_ij_plain() {
        JoinEntry* join_entries;
        cudaMalloc(&join_entries, sizeof(JoinEntry)*lineitem_size);

        const auto kernelStart = std::chrono::high_resolution_clock::now();

        int num_blocks = (part_size + block_size - 1) / block_size;
        ij_lookup_kernel<<<num_blocks, block_size>>>(lineitem_device, lineitem_size, index_structure.device_index, join_entries);
        cudaDeviceSynchronize();

        decltype(output_index) matches;
        cudaError_t error = cudaMemcpyFromSymbol(&matches, output_index, sizeof(matches), 0, cudaMemcpyDeviceToHost);
        assert(error == cudaSuccess);
        //printf("join matches: %u\n", matches);

        num_blocks = (lineitem_size + block_size - 1) / block_size;
        ij_join_kernel<<<num_blocks, block_size>>>(lineitem_device, part_device, join_entries, matches);
        cudaDeviceSynchronize();

        const auto kernelStop = std::chrono::high_resolution_clock::now();
        const auto kernelTime = std::chrono::duration_cast<std::chrono::microseconds>(kernelStop - kernelStart).count()/1000.;
        std::cout << "kernel time: " << kernelTime << " ms\n";
    }

    void compare_join_results(JoinEntry* ref, unsigned ref_size, JoinEntry* actual, unsigned actual_size) {
        std::unordered_map<uint32_t, uint32_t> map;
        for (unsigned i = 0; i < ref_size; ++i) {
            if (map.count(ref[i].lineitem_tid) > 0) {
                std::cerr << "lineitem tid " << ref[i].lineitem_tid << " already in map" << std::endl;
                exit(0);
            }
            map.emplace(ref[i].lineitem_tid, ref[i].part_tid);
        }
        for (unsigned i = 0; i < actual_size; ++i) {
            auto it = map.find(actual[i].lineitem_tid);
            if (it != map.end()) {
                if (it->second != actual[i].part_tid) {
                    std::cerr << "part tid " << actual[i].part_tid << " expected " << it->second << std::endl;
                }
            } else {
                std::cerr << "lineitem tid " << actual[i].lineitem_tid << " not in reference" << std::endl;
            }
        }
    }

    void run_two_phase_ij_buffer_debug() {
        decltype(output_index) matches1 = 0;
        decltype(output_index) matches2 = 0;
        decltype(output_index) zero = 0;

        enum { BLOCK_THREADS = 128, ITEMS_PER_THREAD = 8 }; // TODO optimize

        JoinEntry* join_entries1;
        cudaMallocManaged(&join_entries1, sizeof(JoinEntry)*lineitem_size);

        int num_sms;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        int num_blocks = num_sms*2; // TODO

        const auto start1 = std::chrono::high_resolution_clock::now();
        ij_lookup_kernel_3<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(lineitem_device, lineitem_size, index_structure.device_index, join_entries1);
        cudaDeviceSynchronize();

        cudaError_t error = cudaMemcpyFromSymbol(&matches1, output_index, sizeof(matches1), 0, cudaMemcpyDeviceToHost);
        assert(error == cudaSuccess);
        printf("join matches1: %u\n", matches1);
        printf("debug_cnt: %u\n", debug_cnt);

        error = cudaMemcpyToSymbol(output_index, &zero, sizeof(zero), 0, cudaMemcpyHostToDevice);
        assert(error == cudaSuccess);
        JoinEntry* join_entries2;
        cudaMallocManaged(&join_entries2, sizeof(JoinEntry)*lineitem_size);
        num_blocks = (part_size + block_size - 1) / block_size;
        ij_lookup_kernel<<<num_blocks, block_size>>>(lineitem_device, lineitem_size, index_structure.device_index, join_entries2);
        cudaDeviceSynchronize();

        error = cudaMemcpyFromSymbol(&matches2, output_index, sizeof(matches2), 0, cudaMemcpyDeviceToHost);
        assert(error == cudaSuccess);
        printf("join matches2: %u\n", matches2);

        compare_join_results(join_entries2, matches2, join_entries1, matches1);
        compare_join_results(join_entries1, matches1, join_entries2, matches2);

        const auto d1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start1).count()/1000.;
        std::cout << "kernel time: " << d1 << " ms\n";

        num_blocks = (lineitem_size + block_size - 1) / block_size;

        const auto start2 = std::chrono::high_resolution_clock::now();
        ij_join_kernel<<<num_blocks, block_size>>>(lineitem_device, part_device, join_entries1, matches1);
        cudaDeviceSynchronize();

        const auto kernelStop = std::chrono::high_resolution_clock::now();
        const auto kernelTime = std::chrono::duration_cast<std::chrono::microseconds>(kernelStop - start2).count()/1000.;
        std::cout << "kernel time: " << kernelTime << " ms\n";
    }

    void run_two_phase_ij_buffer() {
        using namespace std;

        decltype(output_index) matches1 = 0;

        enum { BLOCK_THREADS = 256, ITEMS_PER_THREAD = 10 }; // TODO optimize

        JoinEntry* join_entries1;
        cudaMalloc(&join_entries1, sizeof(JoinEntry)*lineitem_size);

        int num_sms;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        int num_blocks = num_sms*4; // TODO

        const auto start1 = std::chrono::high_resolution_clock::now();
        //ij_lookup_kernel<<<num_blocks, BLOCK_THREADS>>>(lineitem_device, lineitem_size, index_structure.device_index, join_entries1);
        //ij_lookup_kernel_2<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(lineitem_device, lineitem_size, index_structure.device_index, join_entries1);
        ij_lookup_kernel_3<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(lineitem_device, lineitem_size, index_structure.device_index, join_entries1);
        //ij_lookup_kernel_4<BLOCK_THREADS><<<num_blocks, BLOCK_THREADS>>>(lineitem_device, lineitem_size, index_structure.device_index, join_entries1);
        cudaDeviceSynchronize();
        const auto d1 = chrono::duration_cast<chrono::microseconds>(std::chrono::high_resolution_clock::now() - start1).count()/1000.;
        std::cout << "kernel time: " << d1 << " ms\n";

        cudaError_t error = cudaMemcpyFromSymbol(&matches1, output_index, sizeof(matches1), 0, cudaMemcpyDeviceToHost);
        assert(error == cudaSuccess);
        printf("join matches1: %u\n", matches1);

        num_blocks = (lineitem_size + block_size - 1) / block_size;

        const auto start2 = std::chrono::high_resolution_clock::now();
        ij_join_kernel<<<num_blocks, block_size>>>(lineitem_device, part_device, join_entries1, matches1);
        cudaDeviceSynchronize();
        const auto kernelStop = std::chrono::high_resolution_clock::now();
        const auto kernelTime = chrono::duration_cast<chrono::microseconds>(kernelStop - start2).count()/1000.;
        std::cout << "kernel time: " << kernelTime << " ms\n";
        std::cout << "complete time: " << d1 + kernelTime << " ms\n";
    }
};

template<class IndexType>
void load_and_run_ij(const std::string& path, bool as_full_pipline_breaker) {
    if (prefetch_index) { throw "not implemented"; }

    helper<IndexType> h;
    h.load_database(path);
    if (as_full_pipline_breaker) {
        printf("full pipline breaker\n");
        h.run_two_phase_ij_buffer();
    } else {
        //h.run_ij();
        h.run_ij_buffer();
    }
}

int main(int argc, char** argv) {
    using namespace std;

    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);// devId);

#ifdef USE_HJ
    if (argc != 2) {
        printf("%s <tpch dataset path>\n", argv[0]);
        return 0;
    }

    helper<int> h;
    h.load_database(argv[1]);
    //std::getc(stdin);
    h.run_hj();
#else
    if (argc < 3) {
        printf("%s <tpch dataset path> <index type: {0: btree, 1: harmonia, 2: radixspline, 3: lowerbound> <1: full pipline breaker>\n", argv[0]);
        return 0;
    }
    enum IndexType : unsigned { btree, harmonia, radixspline, lowerbound, nop } index_type { static_cast<IndexType>(std::stoi(argv[2])) };
    bool full_pipline_breaker = (argc < 4) ? false : std::stoi(argv[3]) != 0;

#ifdef SKIP_SORT
    std::cout << "skip sort step: yes" << std::endl;
#else
    std::cout << "skip sort step: no" << std::endl;
#endif

    switch (index_type) {
        case IndexType::btree: {
            printf("using btree\n");
            using index_type = btree_index<indexed_t, payload_t, device_index_allocator, host_allocator>;
            load_and_run_ij<index_type>(argv[1], full_pipline_breaker);
            break;
        }
        case IndexType::harmonia: {
            printf("using harmonia\n");
            using index_type = harmonia_index<indexed_t, payload_t, device_index_allocator, host_allocator>;
            load_and_run_ij<index_type>(argv[1], full_pipline_breaker);
            break;
        }
        case IndexType::radixspline: {
            printf("using radixspline\n");
            using index_type = radix_spline_index<indexed_t, payload_t, device_index_allocator, host_allocator>;
            load_and_run_ij<index_type>(argv[1], full_pipline_breaker);
            break;
        }
        case IndexType::lowerbound: {
            printf("using lower bound search\n");
            using index_type = lower_bound_index<indexed_t, payload_t, device_index_allocator, host_allocator>;
            load_and_run_ij<index_type>(argv[1], full_pipline_breaker);
            break;
        }
        case IndexType::nop: {
            printf("using no_op_index\n");
            using index_type = no_op_index<indexed_t, payload_t, device_index_allocator, host_allocator>;
            load_and_run_ij<index_type>(argv[1], full_pipline_breaker);
            break;
        }
        default:
            std::cerr << "unknown index type: " << index_type << std::endl;
            return 0;
    }
#endif

/*
    printf("sum1: %lu\n", globalSum1);
    printf("sum2: %lu\n", globalSum2);
*/
    const int64_t result = 100*(globalSum1*1'000)/(globalSum2/1'000);
    printf("%ld.%ld\n", result/1'000'000, result%1'000'000);

    std::cout << std::setprecision(2) << std::scientific
        << "scan_cycles: " << (double)scan_cycles
        << "; sync_cycles: " << (double)sync_cycles
        << "; sort_cycles: " << (double)sort_cycles
        << "; lookup_cycles: " << (double)lookup_cycles 
        << std::endl; 

    cudaDeviceReset();

    return 0;
}
