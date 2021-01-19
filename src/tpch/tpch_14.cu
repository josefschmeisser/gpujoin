#include "common.hpp"

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

#include "LinearProbingHashTable.cuh"
#include "btree.cuh"
#include "btree.cu"

using vector_copy_policy = vector_to_device_array;// vector_to_managed_array;

static constexpr bool prefetch_index = true;

__device__ unsigned int count = 0;
__shared__ bool isLastBlockDone;
__managed__ int tupleCount;

using device_ht_t = LinearProbingHashTable<uint32_t, size_t>::DeviceHandle;

__global__ void hj_build_kernel(size_t n, const part_table_device_t* part, device_ht_t ht) {
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

__global__ void hj_probe_kernel(size_t n, const part_table_device_t* __restrict__ part, const lineitem_table_device_t* __restrict__ lineitem, device_ht_t ht) {
    const char* prefix = "PROMO";
    const uint32_t lower_shipdate = 2449962; // 1995-09-01
    const uint32_t upper_shipdate = 2449992; // 1995-10-01

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

template<class Index>
struct IndexLookup {
    __device__ __forceinline__ btree::payload_t operator() (const btree::key_t key) const;
};

template<>
struct IndexLookup<btree::Node> {
    const btree::Node* tree;
    __device__ __forceinline__ btree::payload_t operator() (const btree::key_t key) const {
        return btree::cuda::btree_lookup_with_hints(tree, key);
    }
};

template<class IndexLookupType>
__global__ void ij_full_kernel(const lineitem_table_device_t* __restrict__ lineitem, const unsigned lineitem_size, const part_table_device_t* __restrict__ part, IndexLookupType index_lookup) {
    const char* prefix = "PROMO";
    const uint32_t lower_shipdate = 2449962; // 1995-09-01
    const uint32_t upper_shipdate = 2449992; // 1995-10-01

    int64_t sum1 = 0;
    int64_t sum2 = 0;

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = index; i < lineitem_size; i += stride) {
        if (lineitem->l_shipdate[i] < lower_shipdate ||
            lineitem->l_shipdate[i] >= upper_shipdate) {
            continue;
        }

        auto payload = index_lookup(lineitem->l_partkey[i]);
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

__global__ void ij_lookup_kernel(const lineitem_table_device_t* __restrict__ lineitem, unsigned lineitem_size, const btree::Node* __restrict__ tree, JoinEntry* __restrict__ join_entries) {
    const uint32_t lower_shipdate = 2449962; // 1995-09-01
    const uint32_t upper_shipdate = 2449992; // 1995-10-01

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = index; i < lineitem_size + 31; i += stride) {
        btree::payload_t payload = btree::invalidTid;
        if (i < lineitem_size &&
            lineitem->l_shipdate[i] >= lower_shipdate &&
            lineitem->l_shipdate[i] < upper_shipdate) {
            payload = btree::cuda::btree_lookup(tree, lineitem->l_partkey[i]);
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

__global__ void ij_join_kernel(const lineitem_table_device_t* __restrict__ lineitem, const part_table_device_t* __restrict__ part, const JoinEntry* __restrict__ join_entries, size_t n) {
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
    if (lane_id() == 0) {
        atomicAdd((unsigned long long int*)&globalSum1, (unsigned long long int)sum1);
        atomicAdd((unsigned long long int*)&globalSum2, (unsigned long long int)sum2);
    }
}


int main(int argc, char** argv) {
    using namespace std;

    if (argc != 3) {
        printf("%s <tpch dataset path> <join method [0-2]>\n", argv[0]);
        return 0;
    }
    enum JoinType : unsigned { HJ, IJ, TWO_PHASE_IJ } join_type { static_cast<unsigned>(std::stoi(argv[2])) };

    Database db;
    load_tables(db, argv[1]);
    const unsigned lineitem_size = db.lineitem.l_orderkey.size();
    const unsigned part_size = db.part.p_partkey.size();

    lineitem_table_device_t* lineitem_device;
    part_table_device_t* part_device;
    {
        auto start = std::chrono::high_resolution_clock::now();
        lineitem_device = copy_relation<vector_copy_policy>(db.lineitem);
        part_device = copy_relation<vector_copy_policy>(db.part);
        auto finish = std::chrono::high_resolution_clock::now();
        auto d = chrono::duration_cast<chrono::milliseconds>(finish - start).count();
        std::cout << "Transfer time: " << d << " ms\n";
    }

    const int blockSize = 32;
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);// devId);

//    int numBlocks = (lineitem_size + blockSize - 1) / blockSize;
    int numBlocks = 32*numSMs;
    printf("numblocks: %d\n", numBlocks);

    decltype(std::chrono::high_resolution_clock::now()) kernelStart, kernelStop;
    switch (join_type) {
        case JoinType::HJ: {
            kernelStart = std::chrono::high_resolution_clock::now();
            LinearProbingHashTable<uint32_t, size_t> ht(part_size);
            hj_build_kernel<<<numBlocks, blockSize>>>(part_size, part_device, ht.deviceHandle);
            hj_probe_kernel<<<numBlocks, blockSize>>>(lineitem_size, part_device, lineitem_device, ht.deviceHandle);
            cudaDeviceSynchronize();
            kernelStop = std::chrono::high_resolution_clock::now();
            break;
        }
        case JoinType::IJ: {
            auto tree = btree::construct(db.part.p_partkey, 0.7);
            btree::prefetchTree(tree, 0);
            IndexLookup<btree::Node> index_lookup {tree};

            kernelStart = std::chrono::high_resolution_clock::now();
            ij_full_kernel<<<numBlocks, blockSize>>>(lineitem_device, lineitem_size, part_device, index_lookup);
            //btree_full_kernel<<<numBlocks, blockSize>>>(lineitem_device, lineitem_size, part_device, tree);
            cudaDeviceSynchronize();
            kernelStop = std::chrono::high_resolution_clock::now();
            break;
        }
        case JoinType::TWO_PHASE_IJ: {
            auto tree = btree::construct(db.part.p_partkey, 0.7);
            btree::prefetchTree(tree, 0);

            JoinEntry* join_entries;
            cudaMalloc(&join_entries, sizeof(JoinEntry)*lineitem_size);

            kernelStart = std::chrono::high_resolution_clock::now();
            ij_lookup_kernel<<<numBlocks, blockSize>>>(lineitem_device, lineitem_size, tree, join_entries);
            cudaDeviceSynchronize();

            decltype(output_index) matches;
            cudaError_t error = cudaMemcpyFromSymbol(&matches, output_index, sizeof(matches), 0, cudaMemcpyDeviceToHost);
            assert(error == cudaSuccess);
            //printf("join matches: %u\n", matches);

            ij_join_kernel<<<numBlocks, blockSize>>>(lineitem_device, part_device, join_entries, matches);
            cudaDeviceSynchronize();
            kernelStop = std::chrono::high_resolution_clock::now();
            break;
        }
        default:
            std::cerr << "unknown join method: " << join_type << std::endl;
    }
    const auto kernelTime = chrono::duration_cast<chrono::microseconds>(kernelStop - kernelStart).count()/1000.;

/*
    printf("sum1: %lu\n", globalSum1);
    printf("sum2: %lu\n", globalSum2);
*/
    const int64_t result = 100*(globalSum1*1'000)/(globalSum2/1'000);
    printf("%ld.%ld\n", result/1'000'000, result%1'000'000);

    const auto finish = std::chrono::high_resolution_clock::now();
    const auto d = chrono::duration_cast<chrono::microseconds>(finish - kernelStart).count()/1000.;
    std::cout << "Kernel time: " << kernelTime << " ms\n";
    std::cout << "Elapsed time with printf: " << d << " ms\n";

    return 0;
}
