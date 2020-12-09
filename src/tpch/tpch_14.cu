#include "common.hpp"

#include <cstddef>
#include <device_atomic_functions.h>
#include <iostream>
#include <limits>
#include <math.h>
#include <cassert>
#include <cstring>
#include <chrono>

#include "LinearProbingHashTable.cuh"

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

__managed__ int64_t globalSum = 0;

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

    const int64_t result = 100*(sum1*1'000)/(sum2/1'000);
 //   atomicAdd((unsigned long long int*)&globalSum, (unsigned long long int)result);
    // TODO proper reduction
}

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

#if USE_PINNED_MEM
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

    // Set a heap size of 128 megabytes. Note that this must
    // be done before any kernel is launched.
    cudaThreadSetLimit(cudaLimitMallocHeapSize, 1024*1024*1024);

    const int blockSize = 32;
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);// devId);

//    int numBlocks = (lineitem_size + blockSize - 1) / blockSize;
    int numBlocks = 32*numSMs;
    printf("numblocks: %d\n", numBlocks);

    auto start = std::chrono::high_resolution_clock::now();

/*
    // char* l_returnflag, char* l_linestatus, int64_t* l_quantity, int64_t* l_extendedprice, int64_t* l_discount, int64_t* l_tax, uint32_t* l_shipdate
    query_1_kernel<<<numBlocks, blockSize>>>(N,
        lineitem.l_returnflag, lineitem.l_linestatus, lineitem.l_quantity, lineitem.l_extendedprice, lineitem.l_discount, lineitem.l_tax, lineitem.l_shipdate);
    cudaDeviceSynchronize();
*/
    LinearProbingHashTable<uint32_t, size_t> ht(part_size);
    build_kernel<<<numBlocks, blockSize>>>(part_size, part_device, ht.deviceHandle);
    probe_kernel<<<numBlocks, blockSize>>>(lineitem_size, part_device, lineitem_device, ht.deviceHandle);
    cudaDeviceSynchronize();

    auto kernelStop = std::chrono::high_resolution_clock::now();
    auto kernelTime = chrono::duration_cast<chrono::microseconds>(kernelStop - start).count()/1000.;

#ifndef NDEBUG
// TODO
#endif

// TODO output

    auto finish = std::chrono::high_resolution_clock::now();
    auto d = chrono::duration_cast<chrono::microseconds>(finish - start).count()/1000.;
    std::cout << "Kernel time: " << kernelTime << " ms\n";
    std::cout << "Elapsed time with printf: " << d << " ms\n";

    return 0;
}
