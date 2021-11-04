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

//#define MEASURE_CYCLES
//#define SKIP_SORT

using namespace cub;

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


template<
    unsigned BLOCK_THREADS,
    unsigned ITEMS_PER_THREAD,
    class    IndexStructureType >
__launch_bounds__ (BLOCK_THREADS)
__global__ void ij_join_scan_kernel(
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

//    using BlockRadixSortT = cub::BlockRadixSort<uint32_t, BLOCK_THREADS, ITEMS_PER_THREAD, uint32_t>;
    using BlockStoreT = cub::BlockStore<uint32_t, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_STORE_WARP_TRANSPOSE>;
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
//        typename BlockRadixSortT::TempStorage temp_storage;
        typename BlockStoreT::TempStorage temp_storage;
    } temp_union;

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

/*
        if (lane_id == 0 && warp_items >= ITEMS_PER_WARP) {
            atomicInc(&fully_occupied_warps, UINT_MAX);
        }
*/
        __syncthreads(); // wait until all threads have gathered enough elements

        BlockStore(temp_storage).Store(d_data, thread_data, warp_items);
        __syncthreads();
    }
}



#if 0 // TODO
template<
    unsigned BLOCK_THREADS,
    unsigned ITEMS_PER_THREAD,
    class    IndexStructureType >
__launch_bounds__ (BLOCK_THREADS)
__global__ void ij_join_finalization_kernel(
    const lineitem_table_plain_t* __restrict__ lineitem,
    const unsigned lineitem_size,
    const part_table_plain_t* __restrict__ part,
    const unsigned part_size,
    const IndexStructureType index_structure,
    int64_t* __restrict__ l_extendedprice_buffer,
    int64_t* __restrict__ l_discount_buffer
    )
{

#ifndef SKIP_SORT

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
printf("warp: %d lane: %d - tid: %u l_partkey: %u\n", warp_id, lane_id, assoc_pos, l_partkey);
            }

            payload_t tid_b = index_structure.cooperative_lookup(active, l_partkey);

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
};

template<class IndexType>
void load_and_run_ij(const std::string& path) {
    if (prefetch_index) { throw "not implemented"; }

    helper<IndexType> h;
    h.load_database(path);
    h.run_ij_buffer();
}

int main(int argc, char** argv) {
    using namespace std;

    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);// devId);

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
        << "; join_cycles: " << (double)join_cycles
        << "; total_cycles: " << (double)total_cycles
        << std::endl; 

    cudaDeviceReset();

    return 0;
}
