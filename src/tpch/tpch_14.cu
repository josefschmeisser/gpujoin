#include "common.hpp"

#include <algorithm>
#include <bits/stdint-uintn.h>
#include <cstddef>
#include <cstdio>
#include <device_atomic_functions.h>
#include <driver_types.h>
#include <iostream>
#include <limits>
#include <math.h>
#include <cassert>
#include <cstring>
#include <chrono>


//#include <cub/block/block_load.cuh>
//#include <cub/block/block_store.cuh>
#include <cub/block/block_radix_sort.cuh>

//#include "thirdparty/cub_test/test_util.h"


#include "LinearProbingHashTable.cuh"
#include "btree.cuh"
#include "btree.cu"
#include "rs.cu"

using namespace cub;

using vector_copy_policy = vector_to_managed_array;
using rs_placement_policy = vector_to_managed_array;

static constexpr bool prefetch_index = false;
static constexpr bool sort_indexed_relation = true;
static constexpr int block_size = 128;
static int num_sms;

const uint32_t lower_shipdate = 2449962; // 1995-09-01
const uint32_t upper_shipdate = 2449992; // 1995-10-01

__device__ unsigned int count = 0;
__shared__ bool isLastBlockDone;
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

#define FULL_MASK 0xffffffff

// see: https://stackoverflow.com/a/44337310
__forceinline__ __device__ unsigned lane_id() {
    unsigned ret;
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

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


struct btree_index {
    const btree::Node* tree_;

    __host__ void construct(const std::vector<btree::key_t>& h_column, const btree::key_t* d_column) {
        auto tree = btree::construct(h_column, 0.7);
        if (prefetch_index) {
            btree::prefetchTree(tree, 0);
        }
        tree_ = tree;
    }

    __device__ __forceinline__ btree::payload_t operator() (const btree::key_t key) const {
        return btree::cuda::btree_lookup(tree_, key);
    //    return btree::cuda::btree_lookup_with_hints(tree_, key); // TODO
    }

    __device__ __forceinline__ btree::payload_t cooperative_lookup(bool active, const btree::key_t key) const {
        return btree::cuda::btree_cooperative_lookup(active, tree_, key);
    }
};

struct radix_spline_index {
    rs::DeviceRadixSpline* d_rs_;
    const btree::key_t* d_column_;

    __host__ void construct(const std::vector<btree::key_t>& h_column, const btree::key_t* d_column) {
        d_column_ = d_column;
        auto h_rs = rs::build_radix_spline(h_column);

        // copy radix spline
        const auto start = std::chrono::high_resolution_clock::now();
        d_rs_ = rs::copy_radix_spline<rs_placement_policy>(h_rs);
        const auto finish = std::chrono::high_resolution_clock::now();
        const auto duration = chrono::duration_cast<chrono::microseconds>(finish - start).count()/1000.;
        std::cout << "radixspline transfer time: " << duration << " ms\n";

        auto rrs __attribute__((unused)) = reinterpret_cast<const rs::RawRadixSpline*>(&h_rs);
        assert(h_column.size() == rrs->num_keys_);
    }

    __device__ __forceinline__ btree::payload_t operator() (const btree::key_t key) const {
        const unsigned estimate = rs::cuda::get_estimate(d_rs_, key);
        const unsigned begin = (estimate < d_rs_->max_error_) ? 0 : (estimate - d_rs_->max_error_);
        const unsigned end = (estimate + d_rs_->max_error_ + 2 > d_rs_->num_keys_) ? d_rs_->num_keys_ : (estimate + d_rs_->max_error_ + 2);

        const auto bound_size = end - begin;
        const unsigned pos = begin + rs::cuda::lower_bound(key, &d_column_[begin], bound_size, [] (const rs::rs_key_t& a, const rs::rs_key_t& b) -> int {
            return a < b;
        });
        return (pos < d_rs_->num_keys_) ? static_cast<btree::payload_t>(pos) : btree::invalidTid;
    }
};

struct lower_bound_index {
    struct device_data_t {
        const btree::key_t* d_column;
        const unsigned d_size;
    }* device_data;

    __host__ void construct(const std::vector<btree::key_t>& h_column, const btree::key_t* d_column) {
        device_data_t tmp { d_column, static_cast<unsigned>(h_column.size()) };
        cudaMalloc(&device_data, sizeof(device_data_t));
        cudaMemcpy(device_data, &tmp, sizeof(device_data_t), cudaMemcpyHostToDevice);
    }

    __device__ __forceinline__ btree::payload_t operator() (const btree::key_t key) const {
//        return btree::cuda::branchy_binary_search(key, device_data->d_column, device_data->d_size);
        return btree::cuda::branch_free_binary_search(key, device_data->d_column, device_data->d_size);
    }
};

using chosen_index_structure = radix_spline_index;// btree_index;

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

        auto payload = index_structure(lineitem->l_partkey[i]);
        if (payload != btree::invalidTid) {
            const size_t part_tid = reinterpret_cast<size_t>(payload);

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
        btree::payload_t payload = btree::invalidTid;
        if (i < lineitem_size &&
            lineitem->l_shipdate[i] >= lower_shipdate &&
            lineitem->l_shipdate[i] < upper_shipdate) {
            payload = index_structure(lineitem->l_partkey[i]);
        }

        int match = payload != btree::invalidTid;
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


template<class T>
__device__ T atomic_sub_safe(T* address, T val) {
    unsigned expected, update, old;
    old = *address;
    do {
        expected = old;
        update = (old > val) ? (old - val) : 0;
        old = atomicCAS(address, expected, update);
    } while (expected != old);
    return old;
}


template<class T>
__forceinline__ __device__ T round_up_pow2(T value) {
    return static_cast<T>(1) << (sizeof(T)*8 - __clz(value - 1));
}

/*
__managed__ unsigned total_scanned = 0;
__managed__ unsigned total_matches = 0;
*/

template<
    int   BLOCK_THREADS,
    int   ITEMS_PER_THREAD,
    class IndexStructureType >
__launch_bounds__ (BLOCK_THREADS)
__global__ void ij_full_kernel_2(
    const lineitem_table_plain_t* __restrict__ lineitem,
    const unsigned lineitem_size,
//    const part_table_plain_t* __restrict__ part,
    IndexStructureType index_structure,
    JoinEntry* __restrict__ join_entries)
{
/*
    const char* prefix = "PROMO";

    int64_t sum1 = 0;
    int64_t sum2 = 0;
*/

    enum {
        MAX_ITEMS_PER_WARP = ITEMS_PER_THREAD * 32,
        WARPS_PER_BLOCK = BLOCK_THREADS / 32,
        // the last summand ensures that each thread can write one more element during the last scan iteration
        BUFFER_SIZE = ITEMS_PER_THREAD*BLOCK_THREADS + BLOCK_THREADS,
        BUFFER_SOFT_LIMIT = ITEMS_PER_THREAD*BLOCK_THREADS
    };
    typedef BlockRadixSort<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;

    __shared__ uint32_t l_partkey_buffer[BUFFER_SIZE];
    __shared__ uint32_t lineitem_tid_buffer[BUFFER_SIZE];
    __shared__ uint32_t buffer_idx;

    __shared__ uint32_t fully_occupied_warps;
    __shared__ uint32_t exhausted_warps;

// TODO
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
    const unsigned tile_size = round_up_pow2((lineitem_size + BLOCK_THREADS - 1) / gridDim.x);
    unsigned tid = blockIdx.x * tile_size + threadIdx.x;
    const unsigned tid_limit = min(tid + tile_size, lineitem_size);

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
                lineitem_tid_buffer[dest_idx] = tid;
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

//            __syncwarp();
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
                p.lineitem_tid = lineitem_tid_buffer[refill_idx_start + prefix_sum];
                p.l_partkey = l_partkey_buffer[refill_idx_start + prefix_sum];
            }

            ideal_refill_cnt -= refill_cnt;
        }

        if (ideal_refill_cnt == 0 && lane_id == 0) {
            atomicInc(&fully_occupied_warps, UINT_MAX);
        }

        __syncthreads(); // wait until all threads have tried to fill their registers

        if (fully_occupied_warps == WARPS_PER_BLOCK) {
            if (warp_id == 0 && lane_id == 0) printf("=== sorting... ===\n");
            assert(join_pairs[0].l_partkey == (join_pairs_raw[0] & FULL_MASK));
/*
            BlockRadixSortT(temp_storage).SortBlockedToStriped(join_pairs_raw, 8, 21); // TODO
*/
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

        uint32_t active_lanes = __ballot_sync(FULL_MASK, local_idx > 0);
        for (unsigned i = 0; active_lanes != 0; ++i) {
            bool active = i < local_idx;
            auto& p = join_pairs[i];
            const auto tid = index_structure.cooperative_lookup(active, p.l_partkey);
//            const auto tid = index_structure(p.l_partkey);

            if (active) {
                assert(tid != btree::invalidTid);
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
            const auto start = std::chrono::high_resolution_clock::now();
            //auto [lineitem_device, lineitem_device_ptrs] = copy_relation<vector_copy_policy>(db.lineitem);
            std::tie(lineitem_device, lineitem_device_ptrs) = copy_relation<vector_copy_policy>(db.lineitem);
            //auto [part_device, part_device_ptrs] = copy_relation<vector_copy_policy>(db.part);
            std::tie(part_device, part_device_ptrs) = copy_relation<vector_copy_policy>(db.part);
            const auto finish = std::chrono::high_resolution_clock::now();
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
        const auto kernelTime = chrono::duration_cast<chrono::microseconds>(kernelStop - kernelStart).count()/1000.;
        std::cout << "kernel time: " << kernelTime << " ms\n";
    }
#endif

    void run_ij() {
        const auto kernelStart = std::chrono::high_resolution_clock::now();

        int num_blocks = (lineitem_size + block_size - 1) / block_size;
        ij_full_kernel<<<num_blocks, block_size>>>(lineitem_device, lineitem_size, part_device, index_structure);
        cudaDeviceSynchronize();

        const auto kernelStop = std::chrono::high_resolution_clock::now();
        const auto kernelTime = chrono::duration_cast<chrono::microseconds>(kernelStop - kernelStart).count()/1000.;
        std::cout << "kernel time: " << kernelTime << " ms\n";
    }

    void run_two_phase_ij_plain() {
        JoinEntry* join_entries;
        cudaMalloc(&join_entries, sizeof(JoinEntry)*lineitem_size);

        const auto kernelStart = std::chrono::high_resolution_clock::now();

        int num_blocks = (part_size + block_size - 1) / block_size;
        ij_lookup_kernel<<<num_blocks, block_size>>>(lineitem_device, lineitem_size, index_structure, join_entries);
        cudaDeviceSynchronize();

        decltype(output_index) matches;
        cudaError_t error = cudaMemcpyFromSymbol(&matches, output_index, sizeof(matches), 0, cudaMemcpyDeviceToHost);
        assert(error == cudaSuccess);
        //printf("join matches: %u\n", matches);

        num_blocks = (lineitem_size + block_size - 1) / block_size;
        ij_join_kernel<<<num_blocks, block_size>>>(lineitem_device, part_device, join_entries, matches);
        cudaDeviceSynchronize();

        const auto kernelStop = std::chrono::high_resolution_clock::now();
        const auto kernelTime = chrono::duration_cast<chrono::microseconds>(kernelStop - kernelStart).count()/1000.;
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
        ij_full_kernel_2<BLOCK_THREADS, ITEMS_PER_THREAD, IndexType><<<num_blocks, BLOCK_THREADS>>>(lineitem_device, lineitem_size, index_structure, join_entries1);
        cudaDeviceSynchronize();

        cudaError_t error = cudaMemcpyFromSymbol(&matches1, output_index, sizeof(matches1), 0, cudaMemcpyDeviceToHost);
        assert(error == cudaSuccess);
        printf("join matches1: %u\n", matches1);

        error = cudaMemcpyToSymbol(output_index, &zero, sizeof(zero), 0, cudaMemcpyHostToDevice);
        assert(error == cudaSuccess);
        JoinEntry* join_entries2;
        cudaMallocManaged(&join_entries2, sizeof(JoinEntry)*lineitem_size);
        num_blocks = (part_size + block_size - 1) / block_size;
        ij_lookup_kernel<<<num_blocks, block_size>>>(lineitem_device, lineitem_size, index_structure, join_entries2);
        cudaDeviceSynchronize();

        error = cudaMemcpyFromSymbol(&matches2, output_index, sizeof(matches2), 0, cudaMemcpyDeviceToHost);
        assert(error == cudaSuccess);
        printf("join matches2: %u\n", matches2);

        compare_join_results(join_entries2, matches2, join_entries1, matches1);
        compare_join_results(join_entries1, matches1, join_entries2, matches2);

        const auto d1 = chrono::duration_cast<chrono::microseconds>(std::chrono::high_resolution_clock::now() - start1).count()/1000.;
        std::cout << "kernel time: " << d1 << " ms\n";

        num_blocks = (lineitem_size + block_size - 1) / block_size;

        const auto start2 = std::chrono::high_resolution_clock::now();
        ij_join_kernel<<<num_blocks, block_size>>>(lineitem_device, part_device, join_entries1, matches1);
        cudaDeviceSynchronize();

        const auto kernelStop = std::chrono::high_resolution_clock::now();
        const auto kernelTime = chrono::duration_cast<chrono::microseconds>(kernelStop - start2).count()/1000.;
        std::cout << "kernel time: " << kernelTime << " ms\n";
    }

    void run_two_phase_ij_buffer() {
        decltype(output_index) matches1 = 0;

        enum { BLOCK_THREADS = 128, ITEMS_PER_THREAD = 8 }; // TODO optimize

        JoinEntry* join_entries1;
        cudaMalloc(&join_entries1, sizeof(JoinEntry)*lineitem_size);

        int num_sms;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        int num_blocks = num_sms*4; // TODO

        const auto start1 = std::chrono::high_resolution_clock::now();
        ij_full_kernel_2<BLOCK_THREADS, ITEMS_PER_THREAD, IndexType><<<num_blocks, BLOCK_THREADS>>>(lineitem_device, lineitem_size, index_structure, join_entries1);
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
    helper<IndexType> h;
    h.load_database(path);
    if (as_full_pipline_breaker) {
        printf("full pipline breaker\n");
        h.run_two_phase_ij_buffer();
    } else {
        h.run_ij();
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

    helper<lower_bound_index> h;
    h.load_database(argv[1]);
    h.run_hj();
#else
    if (argc < 3) {
        printf("%s <tpch dataset path> <index type: {0: btree, 1: radixspline, 2: lowerbound> <1: full pipline breaker>\n", argv[0]);
        return 0;
    }
    enum IndexType : unsigned { btree, radixspline, lowerbound } index_type { static_cast<IndexType>(std::stoi(argv[2])) };
    bool full_pipline_breaker = (argc < 4) ? false : std::stoi(argv[3]) != 0;

    switch (index_type) {
        case IndexType::btree: {
            printf("using btree\n");
            load_and_run_ij<btree_index>(argv[1], full_pipline_breaker);
            break;
        }/* TODO: add cooperative_lookup
        case IndexType::radixspline: {
            printf("using radixspline\n");
            load_and_run_ij<radix_spline_index>(argv[1], full_pipline_breaker);
            break;
        }
        case IndexType::lowerbound: {
            printf("using lower bound search\n");
            load_and_run_ij<lower_bound_index>(argv[1], full_pipline_breaker);
            break;
        }*/
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

    return 0;
}
