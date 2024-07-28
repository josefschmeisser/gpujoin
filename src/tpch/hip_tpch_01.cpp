#include "hip/hip_runtime.h"
#include <bits/stdint-intn.h>
#include <bits/stdint-uintn.h>
#include <cstddef>
//#include <device_atomic_functions.h>
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

// CUDA kernel to add elements of two arrays
__global__
void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

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

__device__ void ht_insert(int32_t k)
{
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

//__managed__ group* globalHT[16];
//__managed__ group* sorted[16];
__device__ group* globalHT[16];
__device__ group* sorted[16];

__device__ unsigned int count = 0;
__device__ bool isLastBlockDone;

//__managed__ uint8_t buffer[1024*1024];
__device__ uint8_t buffer[1024*1024];
__device__ unsigned int groupCount = 0;

//__managed__ int tupleCount;
__device__ int tupleCount;
__device__ uint64_t bufferAddr;


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

        /* Move elements of sorted[0..i-1], that are greater than key, to one position ahead of their current position */
        while (j >= 0 && compare(current, sorted[j])) // sorted[j] > key)
        {
            sorted[j + 1] = sorted[j];
            j = j - 1;
        }
        sorted[j + 1] = current;
    }
}

__global__ void query_1_kernel(
    int n,
    char* __restrict__ l_returnflag,
    char* __restrict__ l_linestatus,
    int64_t* __restrict__ l_quantity,
    int64_t* __restrict__ l_extendedprice,
    int64_t* __restrict__ l_discount,
    int64_t* __restrict__ l_tax,
    uint32_t* __restrict__ l_shipdate)
{
    //constexpr auto threshold_date = to_julian_day(2, 9, 1998); // 1998-09-02
    uint32_t threshold_date = 2451059;

    __shared__ group localGroups[16];
    for (int i = threadIdx.x; i < 16; i += blockDim.x) {
        std::memset(&localGroups[i], 0, sizeof(group));
    }
    __syncthreads();

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        if (l_shipdate[i] > threshold_date) continue;

        uint32_t h = static_cast<uint32_t>(l_returnflag[i]) << 8;
        h |= l_linestatus[i];
        h = hash(h);
        h &= 0b0111;
        group* groupPtr = &localGroups[h];

        if (!groupPtr->in_use) {
            //int atomicCAS(int* address, int compare, int val);
            int stored = atomicCAS(&groupPtr->in_use, 0, 1);
            if (stored == 0) {
                groupPtr->l_returnflag = l_returnflag[i];
                groupPtr->l_linestatus = l_linestatus[i];
                __threadfence();
            }
        }

/* TODO: collision handling
        __syncthreads();
        if (groupPtr->l_returnflag != l_returnflag[i] || groupPtr->l_linestatus != l_linestatus[i]) {
            // TODO handle collisions; spill to global hashtable
            __threadfence();
            printf("first trap\n");
            asm("trap;");
        }
*/

        auto current_l_extendedprice = l_extendedprice[i];
        auto current_l_discount = l_discount[i];
        auto current_l_quantity = l_quantity[i];
        atomicAdd((unsigned long long int*)&groupPtr->sum_qty, (unsigned long long int)current_l_quantity);
        atomicAdd((unsigned long long int*)&groupPtr->sum_base_price, (unsigned long long int)current_l_extendedprice);
        atomicAdd((unsigned long long int*)&groupPtr->sum_disc_price, (unsigned long long int)(current_l_extendedprice * (100 - current_l_discount)));
        atomicAdd((unsigned long long int*)&groupPtr->sum_charge, (unsigned long long int)(current_l_extendedprice * (100 - current_l_discount) * (100 + l_tax[i])));
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
        printf("last\n");

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
//printf("gpu count: %d\n", tupleCount);
        sortGroups(groupCount);

bufferAddr = reinterpret_cast<uint64_t>(&buffer[0]);
//printf("bufferAddr %p = %lu\n", (void*)bufferAddr, bufferAddr);
        count = 0;
    }
}

__global__ void mallocTest()
{
    char* ptr = (char*)malloc(123);
    printf("Thread %d got pointer: %p\n", threadIdx.x, ptr);
    free(ptr);
}

/*
struct lineitem_table_t {
    std::vector<uint32_t> l_orderkey;
    std::vector<uint32_t> l_partkey;
    std::vector<uint32_t> l_suppkey;
    std::vector<uint32_t> l_linenumber;
    std::vector<int64_t> l_quantity;
    std::vector<int64_t> l_extendedprice;
    std::vector<int64_t> l_discount;
    std::vector<int64_t> l_tax;
    std::vector<char> l_returnflag;
    std::vector<char> l_linestatus;
    std::vector<uint32_t> l_shipdate;
    std::vector<uint32_t> l_commitdate;
    std::vector<uint32_t> l_receiptdate;
    std::vector<std::array<char, 25>> l_shipinstruct;
    std::vector<std::array<char, 10>> l_shipmode;
    std::vector<std::string> l_comment;
};
*/

void prepareManaged(lineitem_table_t& src, lineitem_table_plain_t& dst) {
    const auto N = src.l_commitdate.size();

    size_t columnSize = N*sizeof(decltype(src.l_orderkey)::value_type);
    hipMallocManaged(&dst.l_orderkey, columnSize);
    std::memcpy(dst.l_orderkey, src.l_orderkey.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_partkey)::value_type);
    hipMallocManaged(&dst.l_partkey, columnSize);
    std::memcpy(dst.l_partkey, src.l_partkey.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_suppkey)::value_type);
    hipMallocManaged(&dst.l_suppkey, columnSize);
    std::memcpy(dst.l_suppkey, src.l_suppkey.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_linenumber)::value_type);
    hipMallocManaged(&dst.l_linenumber, columnSize);
    std::memcpy(dst.l_linenumber, src.l_linenumber.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_quantity)::value_type);
    hipMallocManaged(&dst.l_quantity, columnSize);
    std::memcpy(dst.l_quantity, src.l_quantity.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_extendedprice)::value_type);
    hipMallocManaged(&dst.l_extendedprice, columnSize);
    std::memcpy(dst.l_extendedprice, src.l_extendedprice.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_discount)::value_type);
    hipMallocManaged(&dst.l_discount, columnSize);
    std::memcpy(dst.l_discount, src.l_discount.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_tax)::value_type);
    hipMallocManaged(&dst.l_tax, columnSize);
    std::memcpy(dst.l_tax, src.l_tax.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_returnflag)::value_type);
    hipMallocManaged(&dst.l_returnflag, columnSize);
    std::memcpy(dst.l_returnflag, src.l_returnflag.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_linestatus)::value_type);
    hipMallocManaged(&dst.l_linestatus, columnSize);
    std::memcpy(dst.l_linestatus, src.l_linestatus.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_shipdate)::value_type);
    hipMallocManaged(&dst.l_shipdate, columnSize);
    std::memcpy(dst.l_shipdate, src.l_shipdate.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_commitdate)::value_type);
    hipMallocManaged(&dst.l_commitdate, columnSize);
    std::memcpy(dst.l_commitdate, src.l_commitdate.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_receiptdate)::value_type);
    hipMallocManaged(&dst.l_receiptdate, columnSize);
    std::memcpy(dst.l_receiptdate, src.l_receiptdate.data(), columnSize);
/*
    columnSize = N*sizeof(decltype(src.l_shipinstruct)::value_type);
    hipMallocManaged(&dst.l_shipinstruct, columnSize);
    std::memcpy(dst.l_shipinstruct, src.l_shipinstruct.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_shipmode)::value_type);
    hipMallocManaged(&dst.l_shipmode, columnSize);
    std::memcpy(dst.l_shipmode, src.l_shipmode.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_comment)::value_type);
    hipMallocManaged(&dst.l_comment, columnSize);
    std::memcpy(dst.l_comment, src.l_comment.data(), columnSize);
*/
}

void prepareDeviceResident(lineitem_table_t& src, lineitem_table_plain_t& dst) {
    const auto N = src.l_commitdate.size();

    size_t columnSize = N*sizeof(decltype(src.l_orderkey)::value_type);
    hipMalloc((void**)&dst.l_orderkey, columnSize);
    hipMemcpy(dst.l_orderkey, src.l_orderkey.data(), columnSize, hipMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_partkey)::value_type);
    hipMalloc((void**)&dst.l_partkey, columnSize);
    hipMemcpy(dst.l_partkey, src.l_partkey.data(), columnSize, hipMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_suppkey)::value_type);
    hipMalloc((void**)&dst.l_suppkey, columnSize);
    hipMemcpy(dst.l_suppkey, src.l_suppkey.data(), columnSize, hipMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_linenumber)::value_type);
    hipMalloc((void**)&dst.l_linenumber, columnSize);
    hipMemcpy(dst.l_linenumber, src.l_linenumber.data(), columnSize, hipMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_quantity)::value_type);
    hipMalloc((void**)&dst.l_quantity, columnSize);
    hipMemcpy(dst.l_quantity, src.l_quantity.data(), columnSize, hipMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_extendedprice)::value_type);
    hipMalloc((void**)&dst.l_extendedprice, columnSize);
    hipMemcpy(dst.l_extendedprice, src.l_extendedprice.data(), columnSize, hipMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_discount)::value_type);
    hipMalloc((void**)&dst.l_discount, columnSize);
    hipMemcpy(dst.l_discount, src.l_discount.data(), columnSize, hipMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_tax)::value_type);
    hipMalloc((void**)&dst.l_tax, columnSize);
    hipMemcpy(dst.l_tax, src.l_tax.data(), columnSize, hipMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_returnflag)::value_type);
    hipMalloc((void**)&dst.l_returnflag, columnSize);
    hipMemcpy(dst.l_returnflag, src.l_returnflag.data(), columnSize, hipMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_linestatus)::value_type);
    hipMalloc((void**)&dst.l_linestatus, columnSize);
    hipMemcpy(dst.l_linestatus, src.l_linestatus.data(), columnSize, hipMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_shipdate)::value_type);
    hipMalloc((void**)&dst.l_shipdate, columnSize);
    hipMemcpy(dst.l_shipdate, src.l_shipdate.data(), columnSize, hipMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_commitdate)::value_type);
    hipMalloc((void**)&dst.l_commitdate, columnSize);
    hipMemcpy(dst.l_commitdate, src.l_commitdate.data(), columnSize, hipMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_receiptdate)::value_type);
    hipMalloc((void**)&dst.l_receiptdate, columnSize);
    hipMemcpy(dst.l_receiptdate, src.l_receiptdate.data(), columnSize, hipMemcpyHostToDevice);

/*
    columnSize = N*sizeof(decltype(src.l_shipinstruct)::value_type);

    columnSize = N*sizeof(decltype(src.l_shipmode)::value_type);

    columnSize = N*sizeof(decltype(src.l_comment)::value_type);
*/
}

int main(int argc, char** argv) {
    using namespace std;

    assert(argc > 1);
    Database db;
    load_tables(db, argv[1]);
    const auto N = db.lineitem.l_commitdate.size();

    lineitem_table_plain_t lineitem;

#if USE_PINNED_MEM
    prepareManaged(db.lineitem, lineitem);
#else
    {
        auto start = std::chrono::high_resolution_clock::now();
        prepareDeviceResident(db.lineitem, lineitem);
        auto finish = std::chrono::high_resolution_clock::now();
        auto d = chrono::duration_cast<chrono::milliseconds>(finish - start).count();
        std::cout << "Transfer time: " << d << " ms\n";
    }
#endif


    // Set a heap size of 128 megabytes. Note that this must
    // be done before any kernel is launched.
    //cudaThreadSetLimit(hipLimitMallocHeapSize, 1024*1024*1024);

/*
    std::memset(globalHT, 0, 16*sizeof(void*));
    std::memset(sorted, 0, 16*sizeof(void*));
    std::memset(buffer, 0, sizeof(buffer));
*/

//hipError_t hipMemset(void *dst, int value, size_t sizeBytes)
    hipMemset(globalHT, 0, 16*sizeof(void*));
    hipMemset(sorted, 0, 16*sizeof(void*));
    hipMemset(buffer, 0, sizeof(buffer));

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    printf("numblocks: %d\n", numBlocks);

    auto start = std::chrono::high_resolution_clock::now();

    // char* l_returnflag, char* l_linestatus, int64_t* l_quantity, int64_t* l_extendedprice, int64_t* l_discount, int64_t* l_tax, uint32_t* l_shipdate
    hipLaunchKernelGGL(query_1_kernel, dim3(numBlocks), dim3(blockSize), 0, 0, N,
        lineitem.l_returnflag, lineitem.l_linestatus, lineitem.l_quantity, lineitem.l_extendedprice, lineitem.l_discount, lineitem.l_tax, lineitem.l_shipdate);
    hipDeviceSynchronize();




//__managed__ group* globalHT[16];
//__managed__ group* sorted[16];
//__managed__ uint8_t buffer[1024*1024];
//__managed__ int tupleCount;
group* hostGlobalHT[16];
group* hostSorted[16];
uint8_t hostBuffer[1024*1024];
int hostTupleCount = 0;

uint64_t gpuOffset;
/*
//hipError_t hipMemcpyDtoH(void *dst, hipDeviceptr_t src, size_t sizeBytes)
hipMemcpyDtoH(hostGlobalHT, globalHT, 16*sizeof(void*));
hipMemcpyDtoH(hostSorted, sorted, 16*sizeof(void*));
hipMemcpyDtoH(hostBuffer, buffer, sizeof(buffer));
hipMemcpyDtoH(&hostTupleCount, &tupleCount
*/
// hipError_t hipMemcpyFromSymbol (void *dst, const void *symbolName, size_t sizeBytes, size_t offset __dparm(0), hipMemcpyKind kind __dparm(hipMemcpyDeviceToHost))
//hipMemcpyFromSymbol(hostGlobalHT, "globalHT", 16*sizeof(void*));
hipMemcpyFromSymbol(hostGlobalHT, HIP_SYMBOL(globalHT), 16*sizeof(void*), 0, hipMemcpyDeviceToHost);
//hipMemcpyFromSymbol(hostSorted, "sorted", 16*sizeof(void*));
hipMemcpyFromSymbol(&hostSorted, HIP_SYMBOL(sorted), 16*sizeof(void*), 0, hipMemcpyDeviceToHost);
//hipMemcpyFromSymbol(hostBuffer, "buffer", sizeof(buffer));
hipMemcpyFromSymbol(&hostBuffer, HIP_SYMBOL(buffer), sizeof(buffer), 0, hipMemcpyDeviceToHost);
//hipMemcpyFromSymbol(&hostTupleCount, "tupleCount", sizeof(int));
hipMemcpyFromSymbol(&hostTupleCount, HIP_SYMBOL(tupleCount), sizeof(int), 0, hipMemcpyDeviceToHost);
printf("hostTupleCount: %d\n", hostTupleCount);

hipMemcpyFromSymbol(&gpuOffset, HIP_SYMBOL(bufferAddr), sizeof(uint64_t), 0, hipMemcpyDeviceToHost);
printf("gpuOffset: %lu\n", gpuOffset);

    const auto hostOffset = reinterpret_cast<ptrdiff_t>(&hostBuffer[0]);

#ifndef NDEBUG
    for (unsigned i = 0; i < 16; i++) {
        if (hostGlobalHT[i] != nullptr) {
            printf("group %d\n", i);
        }
    }

    size_t groupedTupleCount = 0;
    for (unsigned i = 0; i < 16; i++) {
        if (hostGlobalHT[i] == nullptr) continue;

        //auto& t = *hostGlobalHT[i];
        uint8_t* raw = reinterpret_cast<uint8_t*>(hostGlobalHT[i]) - gpuOffset + hostOffset;
        auto& t = *reinterpret_cast<group*>(raw);
        cout << t.l_returnflag << "\t" << t.l_linestatus << "\t" << t.count_order << endl;
        groupedTupleCount += t.count_order;
    }
    printf("groupedTupleCount: %lu\n", groupedTupleCount);
#endif

    for (unsigned i = 0; i < 16; i++) {
        if (hostSorted[i] == nullptr) break;

        //auto& t = *hostSorted[i];
        uint8_t* raw = reinterpret_cast<uint8_t*>(hostSorted[i]) - gpuOffset + hostOffset;
        auto& t = *reinterpret_cast<group*>(raw);
        cout << t.l_returnflag << "\t" << t.l_linestatus << "\t" << t.count_order << endl;
    }

    auto finish = std::chrono::high_resolution_clock::now();
    auto d = chrono::duration_cast<chrono::milliseconds>(finish - start).count();
    std::cout << "Elapsed time with printf: " << d << " ms\n";

    return 0;
}
