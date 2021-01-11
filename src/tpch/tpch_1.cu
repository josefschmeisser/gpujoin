#include <cstddef>
#include <device_atomic_functions.h>
#include <iostream>
#include <limits>
#include <math.h>
#include <cassert>
#include <cstring>
#include <chrono>

#include "common.hpp"

struct group {
    uint64_t sum_qty;
    uint64_t sum_base_price;
    uint64_t sum_disc_price;
    uint64_t sum_charge;
    uint64_t avg_qty;
    uint64_t avg_price;
    uint64_t avg_disc;
    uint64_t count_order;
    char l_returnflag;
    char l_linestatus;
    int in_use;
};

constexpr auto group_size_with_padding = (sizeof(group) + 15) & ~(15);

// 32 bit Murmur3 hash
__device__ uint32_t hash(uint32_t k)
{
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k;
}

/*
-- TPC-H Query 1

select
        l_returnflag,
        l_linestatus,
        sum(l_quantity) as sum_qty,
        sum(l_extendedprice) as sum_base_price,
        sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
        sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
        avg(l_quantity) as avg_qty,
        avg(l_extendedprice) as avg_price,
        avg(l_discount) as avg_disc,
        count(*) as count_order
from
        lineitem
where
        l_shipdate <= date '1998-12-01' - interval '90' day
group by
        l_returnflag,
        l_linestatus
order by
        l_returnflag,
        l_linestatus
*/

__managed__ group* globalHT[16];
__managed__ group* sorted[16];

__device__ unsigned int count = 0;
__shared__ bool isLastBlockDone;

__managed__ uint8_t buffer[1024*1024];
__device__ unsigned int groupCount = 0;

__managed__ int tupleCount;

__device__ group* createGroup() {
/*
    group* ptr = (group*)malloc(sizeof(group));
    assert(ptr);
    printf("Thread %d got pointer: %p\n", threadIdx.x, ptr);
    memset(ptr, 0, sizeof(group));
    return ptr;
*/

    auto old = atomicInc(&groupCount, 0xffffffff);
    group* ptr = reinterpret_cast<group*>(&buffer[old*group_size_with_padding]);
    //printf("Thread %d got pointer: %p\n", threadIdx.x, ptr);
    return ptr;
}

__device__ bool compare(group* a, group* b) {
    return a->l_returnflag < b->l_returnflag || (a->l_returnflag == b->l_returnflag && a->l_linestatus < b->l_linestatus);
}

__device__ void sortGroups(unsigned groupCount) {
    unsigned k = 0;
    for (unsigned i = 0; i < groupCount; ++i) {
        for (; k < 16; ++k) {
            if (globalHT[k] != nullptr) break;
        }

        group* current = globalHT[k++];
        int j = i - 1;

        // insertion sort
        while (j >= 0 && compare(current, sorted[j])) {
            sorted[j + 1] = sorted[j];
            j = j - 1;
        }
        sorted[j + 1] = current;
    }
}

__global__ void query_1_kernel(int n, const lineitem_table_device_t* lineitem) {
    //constexpr auto threshold_date = to_julian_day(2, 9, 1998); // 1998-09-02
    const uint32_t threshold_date = 2451059;

    __shared__ group localGroups[16];
    for (int i = threadIdx.x; i < 16; i += blockDim.x) {
        std::memset(&localGroups[i], 0, sizeof(group));
    }
    __syncthreads();

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        if (lineitem->l_shipdate[i] > threshold_date) continue;

        uint32_t h = static_cast<uint32_t>(lineitem->l_returnflag[i]) << 8;
        h |= lineitem->l_linestatus[i];
        h = hash(h);
        h &= 0b0111;
        group* groupPtr = &localGroups[h];

        if (!groupPtr->in_use) {
            //int atomicCAS(int* address, int compare, int val);
            int stored = atomicCAS(&groupPtr->in_use, 0, 1);
            if (stored == 0) {
                groupPtr->l_returnflag = lineitem->l_returnflag[i];
                groupPtr->l_linestatus = lineitem->l_linestatus[i];
                __threadfence();
            }
        }

/* TODO: collision handling
        __syncthreads();
        if (groupPtr->l_returnflag != lineitem->l_returnflag[i] || groupPtr->l_linestatus != lineitem->l_linestatus[i]) {
            // TODO handle collisions; spill to global hashtable
            __threadfence();
            printf("first trap\n");
            asm("trap;");
        }
*/

        const auto current_l_extendedprice = lineitem->l_extendedprice[i];
        const auto current_l_discount = lineitem->l_discount[i];
        const auto current_l_quantity = lineitem->l_quantity[i];
        atomicAdd((unsigned long long int*)&groupPtr->sum_qty, (unsigned long long int)current_l_quantity);
        atomicAdd((unsigned long long int*)&groupPtr->sum_base_price, (unsigned long long int)current_l_extendedprice);
        atomicAdd((unsigned long long int*)&groupPtr->sum_disc_price, (unsigned long long int)(current_l_extendedprice * (100 - current_l_discount)));
        atomicAdd((unsigned long long int*)&groupPtr->sum_charge, (unsigned long long int)(current_l_extendedprice * (100 - current_l_discount) * (100 + lineitem->l_tax[i])));
        atomicAdd((unsigned long long int*)&groupPtr->avg_qty, (unsigned long long int)current_l_quantity);
        atomicAdd((unsigned long long int*)&groupPtr->avg_price, (unsigned long long int)current_l_extendedprice);
        atomicAdd((unsigned long long int*)&groupPtr->avg_disc, (unsigned long long int)current_l_discount);
        atomicAdd((unsigned long long int*)&groupPtr->count_order, 1ull);
    }

    __syncthreads();
    for (int i = threadIdx.x; i < 16; i += blockDim.x) {
        group* localGroup = &localGroups[i];
        if (!localGroup->in_use) continue;

        group* globalGroup = globalHT[i];

        if (globalGroup == nullptr) {
            group* groupPtr = createGroup();
            groupPtr->l_returnflag = localGroup->l_returnflag;
            groupPtr->l_linestatus = localGroup->l_linestatus;
            auto stored = atomicCAS((unsigned long long int*)&globalHT[i], 0ull, (unsigned long long int)groupPtr);
            if (stored != 0ull) {
                //free(groupPtr); // TODO
                globalGroup = globalHT[i];
            } else {
                globalGroup = groupPtr;
            }
        }

/* TODO: collision handling
        if (localGroup->l_returnflag != globalGroup->l_returnflag || localGroup->l_linestatus != globalGroup->l_linestatus) {
            // TODO
            __threadfence();
            printf("second trap\n");
            asm("trap;");
        }
*/

        atomicAdd((unsigned long long int*)&globalGroup->sum_qty, (unsigned long long int)localGroup->sum_qty);
        atomicAdd((unsigned long long int*)&globalGroup->sum_base_price, (unsigned long long int)localGroup->sum_base_price);
        atomicAdd((unsigned long long int*)&globalGroup->sum_disc_price, (unsigned long long int)localGroup->sum_disc_price);
        atomicAdd((unsigned long long int*)&globalGroup->sum_charge, (unsigned long long int)localGroup->sum_charge);
        atomicAdd((unsigned long long int*)&globalGroup->avg_qty, (unsigned long long int)localGroup->avg_qty);
        atomicAdd((unsigned long long int*)&globalGroup->avg_price, (unsigned long long int)localGroup->avg_price);
        atomicAdd((unsigned long long int*)&globalGroup->avg_disc, (unsigned long long int)localGroup->avg_disc);
        atomicAdd((unsigned long long int*)&globalGroup->count_order, (unsigned long long int)localGroup->count_order);
    }

    __threadfence();

    // see CUDA guide: B.5. Memory Fence Functions
    if (threadIdx.x == 0) {
        // Thread 0 signals that it is done.
        unsigned int value = atomicInc(&count, gridDim.x);
        // Thread 0 determines if its block is the last
        // block to be done.
        isLastBlockDone = (value == (gridDim.x - 1));
    }

    __syncthreads();

    if (isLastBlockDone && threadIdx.x == 0) {
        //printf("last\n");

        tupleCount = 0;
        unsigned groupCount = 0;

        for (int i = 0; i < 16; ++i) {
            group* globalGroup = globalHT[i];
            if (globalGroup == nullptr) { continue; }
            //printf("group: %p\n", globalGroup);

            // TODO adjust decimal point
            globalGroup->avg_qty /= globalGroup->count_order;
            globalGroup->avg_price /= globalGroup->count_order;
            globalGroup->avg_disc /= globalGroup->count_order;

            tupleCount += globalGroup->count_order;
            groupCount++;
        }

        sortGroups(groupCount);

        count = 0;
    }
}

int main(int argc, char** argv) {
    using namespace std;

    assert(argc > 1);
    Database db;
    load_tables(db, argv[1]);
    const auto N = db.lineitem.l_commitdate.size();

    lineitem_table_device_t* lineitem;
#if USE_PINNED_MEM
    //prepareManaged(db.lineitem, lineitem);
    lineitem = copy_relation<vector_to_managed_array>(db.lineitem);
#else
    {
        auto start = std::chrono::high_resolution_clock::now();
        lineitem = copy_relation<vector_to_device_array>(db.lineitem);
        auto finish = std::chrono::high_resolution_clock::now();
        auto d = chrono::duration_cast<chrono::milliseconds>(finish - start).count();
        std::cout << "Transfer time: " << d << " ms\n";
    }
#endif

    // Set a heap size of 128 megabytes. Note that this must
    // be done before any kernel is launched.
    cudaThreadSetLimit(cudaLimitMallocHeapSize, 1024*1024*1024);

    std::memset(globalHT, 0, 16*sizeof(void*));
    std::memset(sorted, 0, 16*sizeof(void*));
    std::memset(buffer, 0, sizeof(buffer));

    int blockSize = 32;
//    int numBlocks = (N + blockSize - 1) / blockSize;
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);// devId);
    int numBlocks = 32*numSMs;
    printf("numblocks: %d\n", numBlocks);

    auto start = std::chrono::high_resolution_clock::now();

    query_1_kernel<<<numBlocks, blockSize>>>(N, lineitem);
    cudaDeviceSynchronize();

    auto kernelStop = std::chrono::high_resolution_clock::now();
    auto kernelTime = chrono::duration_cast<chrono::microseconds>(kernelStop - start).count()/1000.;

#ifndef NDEBUG
    for (unsigned i = 0; i < 16; i++) {
        if (globalHT[i] != nullptr) {
            printf("group %d\n", i);
        }
    }
    printf("device tupleCount: %d\n", tupleCount);

    size_t hostTupleCount = 0;
    for (unsigned i = 0; i < 16; i++) {
        if (globalHT[i] == nullptr) continue;

        //printf("%p\n", sorted[i]);
        auto& t = *globalHT[i];
        cout << t.l_returnflag << "\t" << t.l_linestatus << "\t" << t.count_order << endl;
        hostTupleCount += t.count_order;
    }
    printf("hostTupleCount: %lu\n", hostTupleCount);
#endif

    for (unsigned i = 0; i < 16; i++) {
        if (sorted[i] == nullptr) break;

        auto& t = *sorted[i];
        cout << t.l_returnflag << "\t" << t.l_linestatus << "\t" << t.count_order << endl;
    }

    auto finish = std::chrono::high_resolution_clock::now();
    auto d = chrono::duration_cast<chrono::microseconds>(finish - start).count()/1000.;
    std::cout << "Kernel time: " << kernelTime << " ms\n";
    std::cout << "Elapsed time with printf: " << d << " ms\n";

    return 0;
}
