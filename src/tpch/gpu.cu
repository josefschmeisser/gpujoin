#include <bits/stdint-intn.h>
#include <bits/stdint-uintn.h>
#include <cstddef>
#include <device_atomic_functions.h>
#include <iostream>
#include <limits>
#include <math.h>
#include <cassert>
#include <cstring>

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

__managed__ group* globalHT[16];

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
    printf("Thread %d got pointer: %p\n", threadIdx.x, ptr);
    return ptr;
}

__global__
void query_1_kernel(int n, char* l_returnflag, char* l_linestatus, int64_t* l_quantity, int64_t* l_extendedprice, int64_t* l_discount, int64_t* l_tax, uint32_t* l_shipdate)
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
//                free(groupPtr); // TODO
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

/*
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

        for (int i = 0; i < 16; i += ++i) {
            group* globalGroup = globalHT[i];
            if (globalGroup == nullptr) { continue; }

            // TODO adjust decimal point
            globalGroup->avg_qty /= globalGroup->count_order;
            globalGroup->avg_price /= globalGroup->count_order;
            globalGroup->avg_disc /= globalGroup->count_order;

            tupleCount += globalGroup->count_order;
        }

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

int main(int argc, char** argv) {
    assert(argc > 1);
    Database db;
    load_tables(db, argv[1]);


    const auto N = db.lineitem.l_commitdate.size();
    lineitem_table_mgd_t lineitem;

    size_t columnSize = N*sizeof(decltype(db.lineitem.l_orderkey)::value_type);
    cudaMallocManaged(&lineitem.l_orderkey, columnSize);
    std::memcpy(lineitem.l_orderkey, db.lineitem.l_orderkey.data(), columnSize);

    columnSize = N*sizeof(decltype(db.lineitem.l_partkey)::value_type);
    cudaMallocManaged(&lineitem.l_partkey, columnSize);
    std::memcpy(lineitem.l_partkey, db.lineitem.l_partkey.data(), columnSize);

    columnSize = N*sizeof(decltype(db.lineitem.l_suppkey)::value_type);
    cudaMallocManaged(&lineitem.l_suppkey, columnSize);
    std::memcpy(lineitem.l_suppkey, db.lineitem.l_suppkey.data(), columnSize);

    columnSize = N*sizeof(decltype(db.lineitem.l_linenumber)::value_type);
    cudaMallocManaged(&lineitem.l_linenumber, columnSize);
    std::memcpy(lineitem.l_linenumber, db.lineitem.l_linenumber.data(), columnSize);

    columnSize = N*sizeof(decltype(db.lineitem.l_quantity)::value_type);
    cudaMallocManaged(&lineitem.l_quantity, columnSize);
    std::memcpy(lineitem.l_quantity, db.lineitem.l_quantity.data(), columnSize);

    columnSize = N*sizeof(decltype(db.lineitem.l_extendedprice)::value_type);
    cudaMallocManaged(&lineitem.l_extendedprice, columnSize);
    std::memcpy(lineitem.l_extendedprice, db.lineitem.l_extendedprice.data(), columnSize);

    columnSize = N*sizeof(decltype(db.lineitem.l_discount)::value_type);
    cudaMallocManaged(&lineitem.l_discount, columnSize);
    std::memcpy(lineitem.l_discount, db.lineitem.l_discount.data(), columnSize);

    columnSize = N*sizeof(decltype(db.lineitem.l_tax)::value_type);
    cudaMallocManaged(&lineitem.l_tax, columnSize);
    std::memcpy(lineitem.l_tax, db.lineitem.l_tax.data(), columnSize);

    columnSize = N*sizeof(decltype(db.lineitem.l_returnflag)::value_type);
    cudaMallocManaged(&lineitem.l_returnflag, columnSize);
    std::memcpy(lineitem.l_returnflag, db.lineitem.l_returnflag.data(), columnSize);

    columnSize = N*sizeof(decltype(db.lineitem.l_linestatus)::value_type);
    cudaMallocManaged(&lineitem.l_linestatus, columnSize);
    std::memcpy(lineitem.l_linestatus, db.lineitem.l_linestatus.data(), columnSize);

    columnSize = N*sizeof(decltype(db.lineitem.l_shipdate)::value_type);
    cudaMallocManaged(&lineitem.l_shipdate, columnSize);
    std::memcpy(lineitem.l_shipdate, db.lineitem.l_shipdate.data(), columnSize);

    columnSize = N*sizeof(decltype(db.lineitem.l_commitdate)::value_type);
    cudaMallocManaged(&lineitem.l_commitdate, columnSize);
    std::memcpy(lineitem.l_commitdate, db.lineitem.l_commitdate.data(), columnSize);

    columnSize = N*sizeof(decltype(db.lineitem.l_receiptdate)::value_type);
    cudaMallocManaged(&lineitem.l_receiptdate, columnSize);
    std::memcpy(lineitem.l_receiptdate, db.lineitem.l_receiptdate.data(), columnSize);
/*
    columnSize = N*sizeof(decltype(db.lineitem.l_shipinstruct)::value_type);
    cudaMallocManaged(&lineitem.l_shipinstruct, columnSize);
    std::memcpy(lineitem.l_shipinstruct, db.lineitem.l_shipinstruct.data(), columnSize);

    columnSize = N*sizeof(decltype(db.lineitem.l_shipmode)::value_type);
    cudaMallocManaged(&lineitem.l_shipmode, columnSize);
    std::memcpy(lineitem.l_shipmode, db.lineitem.l_shipmode.data(), columnSize);

    columnSize = N*sizeof(decltype(db.lineitem.l_comment)::value_type);
    cudaMallocManaged(&lineitem.l_comment, columnSize);
    std::memcpy(lineitem.l_comment, db.lineitem.l_comment.data(), columnSize);
*/

    // Set a heap size of 128 megabytes. Note that this must
    // be done before any kernel is launched.
    cudaThreadSetLimit(cudaLimitMallocHeapSize, 1024*1024*1024);
#if 0
    mallocTest<<<1, 5>>>();
    cudaThreadSynchronize();
    return 0;

    int N = 1<<20;
    float *x, *y;

    // Allocate Unified Memory -- accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Launch kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);
#endif

    std::memset(globalHT, 0, 16*sizeof(void*));
    std::memset(buffer, 0, sizeof(buffer));

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    printf("numblocks: %d\n", numBlocks);
    // char* l_returnflag, char* l_linestatus, int64_t* l_quantity, int64_t* l_extendedprice, int64_t* l_discount, int64_t* l_tax, uint32_t* l_shipdate
    query_1_kernel<<<numBlocks, blockSize>>>(N,
        lineitem.l_returnflag, lineitem.l_linestatus, lineitem.l_quantity, lineitem.l_extendedprice, lineitem.l_discount, lineitem.l_tax, lineitem.l_shipdate);

    cudaDeviceSynchronize();
    for (unsigned i = 0; i < 16; i++) {
        if (globalHT[i] != nullptr) {
            printf("group %d\n", i);
        }
    }
    printf("device tupleCount: %d\n", tupleCount);

    using namespace std;
    size_t hostTupleCount = 0;
    for (unsigned i = 0; i < 16; i++) {
        if (globalHT[i] == nullptr) continue;

        //printf("%p\n", sorted[i]);
        auto& t = *globalHT[i];
        cout << t.l_returnflag << "\t" << t.l_linestatus << "\t" << t.count_order << endl;
        hostTupleCount += t.count_order;
    }
    printf("hostTupleCount: %lu\n", hostTupleCount);

    return 0;
}
