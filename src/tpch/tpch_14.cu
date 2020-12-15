#include "common.hpp"

#include <cstddef>
#include <cstdio>
#include <device_atomic_functions.h>
#include <iostream>
#include <limits>
#include <math.h>
#include <cassert>
#include <cstring>
#include <chrono>

#include "LinearProbingHashTable.cuh"
#include "btree.cuh"

__device__ unsigned int count = 0;
__shared__ bool isLastBlockDone;
__managed__ int tupleCount;

using device_ht_t = LinearProbingHashTable<uint32_t, size_t>::DeviceHandle;

__global__ void build_kernel(size_t n, part_table_device_t* part, device_ht_t ht) {
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
        } else if (str_a[i] != str_b[i]){
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

__global__ void probe_kernel(size_t n, part_table_device_t* part, lineitem_table_device_t* lineitem, device_ht_t ht) {
    const char* prefix = "PROMO";
    const uint32_t lower_shipdate = 2449962; // 1995-09-01
    const uint32_t upper_shipdate = 2449992; // 1995-10-01

    int64_t sum1 = 0;
    int64_t sum2 = 0;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
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
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum1 += __shfl_down_sync(FULL_MASK, sum1, offset);
        sum2 += __shfl_down_sync(FULL_MASK, sum2, offset);
    }
    if (lane_id() == 0) {
        atomicAdd((unsigned long long int*)&globalSum1, (unsigned long long int)sum1);
        atomicAdd((unsigned long long int*)&globalSum2, (unsigned long long int)sum2);
    }
}

__global__ void btree_kernel(lineitem_table_device_t* lineitem, unsigned lineitem_size, part_table_device_t* part, btree::Node* tree) {
    const char* prefix = "PROMO";
    const uint32_t lower_shipdate = 2449962; // 1995-09-01
    const uint32_t upper_shipdate = 2449992; // 1995-10-01

    int64_t sum1 = 0;
    int64_t sum2 = 0;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < lineitem_size; i += stride) {
        if (lineitem->l_shipdate[i] < lower_shipdate ||
            lineitem->l_shipdate[i] >= upper_shipdate) {
            continue;
        }

        auto payload = btree::cuda::btree_lookup(tree, lineitem->l_partkey[i]);
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

/*
inline __device__ unsigned __funnelshift_l(unsigned low32, unsigned high32, unsigned shiftWidth) {
  unsigned result;
  asm("shf.l.wrap.b32 %0, %1, %2, %3;"
      : "=r"(result)
      : "r"(low32), "r"(high32), "r"(shiftWidth));
  return result;
}*/

__global__ void btree_lookup_kernel(lineitem_table_device_t* lineitem, unsigned lineitem_size, btree::Node* tree, JoinEntry* join_entries) {
//    const char* prefix = "PROMO";
    const uint32_t lower_shipdate = 2449962; // 1995-09-01
    const uint32_t upper_shipdate = 2449992; // 1995-10-01

//    unsigned matches = 0;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < lineitem_size; i += stride) { // FIXME
    printf("i: %d\n", i);
    /*
        if (lineitem->l_shipdate[i] < lower_shipdate ||
            lineitem->l_shipdate[i] >= upper_shipdate) {
            continue;
        }*/

        auto payload = btree::cuda::btree_lookup(tree, lineitem->l_partkey[i]);
        printf("lookup %lu\n", payload);
        /*
        if (payload != btree::invalidTid) {
            unsigned base = atomicAggInc(output_index);
            __syncwarp();

//            unsigned lane = lane_id();
            // _device__ ​ unsigned int __funnelshift_l ( unsigned int  lo, unsigned int  hi, unsigned int  shift ) 
            unsigned mask = __funnelshift_l(0xffffffff, 0, lane_id());

        }*/

        int match = payload != btree::invalidTid;
        unsigned my_lane = lane_id();
        unsigned mask = __ballot_sync(FULL_MASK, match);
        unsigned right = __funnelshift_l(0xffffffff, 0, my_lane);
        printf("right %lu\n", right);
        unsigned offset = __popc(mask & right);
        
        printf("lane: %u offset: %u\n", my_lane, offset);

        unsigned base = 0;
        if (my_lane == 0) {
            base = atomicInc(&output_index, __popc(mask));
        }

        //T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize);
        base = __shfl_sync(FULL_MASK, base, 0);

        join_entries[base + offset].lineitem_tid = i;
    }
}

#if 0
__global__ void join_kernel(lineitem_table_device_t* lineitem, unsigned linteitem_size, part_table_device_t* part, btree::Node* tree) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < linteitem_size; i += stride) {
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
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum1 += __shfl_down_sync(FULL_MASK, sum1, offset);
        sum2 += __shfl_down_sync(FULL_MASK, sum2, offset);
    }
    if (lane_id() == 0) {
        atomicAdd((unsigned long long int*)&globalSum1, (unsigned long long int)sum1);
        atomicAdd((unsigned long long int*)&globalSum2, (unsigned long long int)sum2);
    }
}
#endif


int main(int argc, char** argv) {
    using namespace std;

    assert(argc > 1);
    Database db;
    load_tables(db, argv[1]);
    const auto lineitem_size = db.lineitem.l_orderkey.size();
    const auto part_size = db.part.p_partkey.size();

    lineitem_table_device_t* lineitem_device;
    cudaMallocManaged(&lineitem_device, sizeof(lineitem_table_device_t));
    part_table_device_t* part_device;
    cudaMallocManaged(&part_device, sizeof(part_table_device_t));

#define USE_PINNED_MEM
#ifdef USE_PINNED_MEM
    copy_relation<vector_to_managed_array>(db.lineitem, *lineitem_device);
    copy_relation<vector_to_managed_array>(db.part, *part_device);
#else
    {
        auto start = std::chrono::high_resolution_clock::now();
        copy_relation<vector_to_device_array>(db.lineitem, *lineitem_device);
        copy_relation<vector_to_device_array>(db.part, *part_device);
        auto finish = std::chrono::high_resolution_clock::now();
        auto d = chrono::duration_cast<chrono::milliseconds>(finish - start).count();
        std::cout << "Transfer time: " << d << " ms\n";
    }
#endif

    //cudaThreadSetLimit(cudaLimitMallocHeapSize, 1024*1024*1024);

    const int blockSize = 32;
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);// devId);

//    int numBlocks = (lineitem_size + blockSize - 1) / blockSize;
    int numBlocks = 32*numSMs;
    printf("numblocks: %d\n", numBlocks);

//#define HJ_QUERY
#ifdef HJ_QUERY
    auto start = std::chrono::high_resolution_clock::now();
    LinearProbingHashTable<uint32_t, size_t> ht(part_size);
    build_kernel<<<numBlocks, blockSize>>>(part_size, part_device, ht.deviceHandle);
    probe_kernel<<<numBlocks, blockSize>>>(lineitem_size, part_device, lineitem_device, ht.deviceHandle);
    cudaDeviceSynchronize();
#else
    auto tree = btree::construct(db.part.p_partkey, 0.7);
    btree::prefetchTree(tree, 0);
    auto start = std::chrono::high_resolution_clock::now();
//    btree_kernel<<<numBlocks, blockSize>>>(lineitem_device, lineitem_size, part_device, tree);


//__global__ void btree_lookup_kernel(lineitem_table_device_t* lineitem, unsigned linteitem_size, btree::Node* tree, JoinEntry* join_entries)
JoinEntry* join_entries;
cudaMalloc(&join_entries, sizeof(JoinEntry)*lineitem_size);
btree_lookup_kernel<<<1, 32>>>(lineitem_device, 128, tree, join_entries);

    cudaDeviceSynchronize();
return;
#endif

    auto kernelStop = std::chrono::high_resolution_clock::now();
    auto kernelTime = chrono::duration_cast<chrono::microseconds>(kernelStop - start).count()/1000.;

#ifndef NDEBUG
// TODO
#endif

printf("sum1: %lu\n", globalSum1);
printf("sum2: %lu\n", globalSum2);
    int64_t result = 100*(globalSum1*1'000)/(globalSum2/1'000);
    printf("%ld.%ld\n", result/1'000'000, result%1'000'000);

    auto finish = std::chrono::high_resolution_clock::now();
    auto d = chrono::duration_cast<chrono::microseconds>(finish - start).count()/1000.;
    std::cout << "Kernel time: " << kernelTime << " ms\n";
    std::cout << "Elapsed time with printf: " << d << " ms\n";

    return 0;
}
