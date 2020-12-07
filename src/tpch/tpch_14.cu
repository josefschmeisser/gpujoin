#include "common.hpp"

#include <bits/stdint-intn.h>
#include <cstddef>
#include <device_atomic_functions.h>
#include <iostream>
#include <limits>
#include <math.h>
#include <cassert>
#include <cstring>
#include <chrono>

#include "LinearProbingHashTable.cuh"

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

__managed__ group* globalHT[16];
__managed__ group* sorted[16];

__device__ unsigned int count = 0;
__shared__ bool isLastBlockDone;

__managed__ uint8_t buffer[1024*1024];
__device__ unsigned int groupCount = 0;

__managed__ int tupleCount;

__device__ bool compare(group* a, group* b) {
    return a->l_returnflag < b->l_returnflag || (a->l_returnflag == b->l_returnflag && a->l_linestatus < b->l_linestatus);
}


/*
void query_14_part_build(Database& db) {
    int64_t sum1 = 0;
    int64_t sum2 = 0;

    constexpr std::string_view prefix = "PROMO";
    constexpr auto lower_shipdate = to_julian_day(1, 9, 1995); // 1995-09-01
    constexpr auto upper_shipdate = to_julian_day(1, 10, 1995); // 1995-10-01

    auto& part = db.part;
    auto& lineitem = db.lineitem;

    std::unordered_map<uint32_t, size_t> ht(part.p_partkey.size());
    for (size_t i = 0; i < part.p_partkey.size(); ++i) {
        ht.emplace(part.p_partkey[i], i);
    }

    // aggregation loop
    for (size_t i = 0; i < lineitem.l_partkey.size(); ++i) {
        if (lineitem.l_shipdate[i] < lower_shipdate ||
            lineitem.l_shipdate[i] >= upper_shipdate) {
            continue;
        }

        // probe
        auto it = ht.find(lineitem.l_partkey[i]);
        if (it == ht.end()) {
            continue;
        }
        size_t j = it->second;

        auto extendedprice = lineitem.l_extendedprice[i];
        auto discount = lineitem.l_discount[i];
        auto summand = extendedprice * (100 - discount);
        sum2 += summand;

        auto& type = part.p_type[j];
        if (std::strncmp(type.data(), prefix.data(), prefix.size()) == 0) {
            sum1 += summand;
        }
    }

    sum1 *= 1'000;
    sum2 /= 1'000;
    int64_t result = 100*sum1/sum2;
    printf("%ld.%ld\n", result/1'000'000, result%1'000'000);
}
*/

__global__ void build_kernel(size_t n, part_table_device_t* part, LinearProbingHashTable::DeviceHandle ht) {
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
            if ((int)str_a[i] - (int)str_b[i]) < 0) match = 0 - (i + 1);
        }
        i++;
    }
    return match;
}

__managed__ int64_t globalSum = 0;

__global__ void probe_kernel(size_t n, lineitem_table_device_t* lineitem, LinearProbingHashTable::DeviceHandle ht) {
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

        size_t tid;
        bool match = ht.lookup(lineitem->l_partkey[i], tid);
        if (match) {
            const auto extendedprice = lineitem->l_lineitem[tid];
            const auto discount = lineitem->l_discount[tid];
            const auto summand = extendedprice * (100 - discount);
            sum2 += summand;

            if (std::strncmp(lineitem->l_type[tid], prefix, 5) == 0) {
                sum1 += summand;
            }
        }
    }

    const int64_t result = 100*(sum1*1'000)/(sum2/1'000);
    atomicAdd((unsigned long long int*)&globalSum, (unsigned long long int)result);
    // TODO proper reduction
}

void prepareManaged(lineitem_table_t& src, lineitem_table_device_t& dst) {
    const auto N = src.l_commitdate.size();

    size_t columnSize = N*sizeof(decltype(src.l_orderkey)::value_type);
    cudaMallocManaged(&dst.l_orderkey, columnSize);
    std::memcpy(dst.l_orderkey, src.l_orderkey.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_partkey)::value_type);
    cudaMallocManaged(&dst.l_partkey, columnSize);
    std::memcpy(dst.l_partkey, src.l_partkey.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_suppkey)::value_type);
    cudaMallocManaged(&dst.l_suppkey, columnSize);
    std::memcpy(dst.l_suppkey, src.l_suppkey.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_linenumber)::value_type);
    cudaMallocManaged(&dst.l_linenumber, columnSize);
    std::memcpy(dst.l_linenumber, src.l_linenumber.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_quantity)::value_type);
    cudaMallocManaged(&dst.l_quantity, columnSize);
    std::memcpy(dst.l_quantity, src.l_quantity.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_extendedprice)::value_type);
    cudaMallocManaged(&dst.l_extendedprice, columnSize);
    std::memcpy(dst.l_extendedprice, src.l_extendedprice.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_discount)::value_type);
    cudaMallocManaged(&dst.l_discount, columnSize);
    std::memcpy(dst.l_discount, src.l_discount.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_tax)::value_type);
    cudaMallocManaged(&dst.l_tax, columnSize);
    std::memcpy(dst.l_tax, src.l_tax.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_returnflag)::value_type);
    cudaMallocManaged(&dst.l_returnflag, columnSize);
    std::memcpy(dst.l_returnflag, src.l_returnflag.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_linestatus)::value_type);
    cudaMallocManaged(&dst.l_linestatus, columnSize);
    std::memcpy(dst.l_linestatus, src.l_linestatus.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_shipdate)::value_type);
    cudaMallocManaged(&dst.l_shipdate, columnSize);
    std::memcpy(dst.l_shipdate, src.l_shipdate.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_commitdate)::value_type);
    cudaMallocManaged(&dst.l_commitdate, columnSize);
    std::memcpy(dst.l_commitdate, src.l_commitdate.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_receiptdate)::value_type);
    cudaMallocManaged(&dst.l_receiptdate, columnSize);
    std::memcpy(dst.l_receiptdate, src.l_receiptdate.data(), columnSize);
/*
    columnSize = N*sizeof(decltype(src.l_shipinstruct)::value_type);
    cudaMallocManaged(&dst.l_shipinstruct, columnSize);
    std::memcpy(dst.l_shipinstruct, src.l_shipinstruct.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_shipmode)::value_type);
    cudaMallocManaged(&dst.l_shipmode, columnSize);
    std::memcpy(dst.l_shipmode, src.l_shipmode.data(), columnSize);

    columnSize = N*sizeof(decltype(src.l_comment)::value_type);
    cudaMallocManaged(&dst.l_comment, columnSize);
    std::memcpy(dst.l_comment, src.l_comment.data(), columnSize);
*/
}

void prepareDeviceResident(lineitem_table_t& src, lineitem_table_device_t& dst) {
    const auto N = src.l_commitdate.size();

    size_t columnSize = N*sizeof(decltype(src.l_orderkey)::value_type);
    cudaMalloc((void**)&dst.l_orderkey, columnSize);
    cudaMemcpy(dst.l_orderkey, src.l_orderkey.data(), columnSize, cudaMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_partkey)::value_type);
    cudaMalloc((void**)&dst.l_partkey, columnSize);
    cudaMemcpy(dst.l_partkey, src.l_partkey.data(), columnSize, cudaMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_suppkey)::value_type);
    cudaMalloc((void**)&dst.l_suppkey, columnSize);
    cudaMemcpy(dst.l_suppkey, src.l_suppkey.data(), columnSize, cudaMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_linenumber)::value_type);
    cudaMalloc((void**)&dst.l_linenumber, columnSize);
    cudaMemcpy(dst.l_linenumber, src.l_linenumber.data(), columnSize, cudaMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_quantity)::value_type);
    cudaMalloc((void**)&dst.l_quantity, columnSize);
    cudaMemcpy(dst.l_quantity, src.l_quantity.data(), columnSize, cudaMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_extendedprice)::value_type);
    cudaMalloc((void**)&dst.l_extendedprice, columnSize);
    cudaMemcpy(dst.l_extendedprice, src.l_extendedprice.data(), columnSize, cudaMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_discount)::value_type);
    cudaMalloc((void**)&dst.l_discount, columnSize);
    cudaMemcpy(dst.l_discount, src.l_discount.data(), columnSize, cudaMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_tax)::value_type);
    cudaMalloc((void**)&dst.l_tax, columnSize);
    cudaMemcpy(dst.l_tax, src.l_tax.data(), columnSize, cudaMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_returnflag)::value_type);
    cudaMalloc((void**)&dst.l_returnflag, columnSize);
    cudaMemcpy(dst.l_returnflag, src.l_returnflag.data(), columnSize, cudaMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_linestatus)::value_type);
    cudaMalloc((void**)&dst.l_linestatus, columnSize);
    cudaMemcpy(dst.l_linestatus, src.l_linestatus.data(), columnSize, cudaMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_shipdate)::value_type);
    cudaMalloc((void**)&dst.l_shipdate, columnSize);
    cudaMemcpy(dst.l_shipdate, src.l_shipdate.data(), columnSize, cudaMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_commitdate)::value_type);
    cudaMalloc((void**)&dst.l_commitdate, columnSize);
    cudaMemcpy(dst.l_commitdate, src.l_commitdate.data(), columnSize, cudaMemcpyHostToDevice);

    columnSize = N*sizeof(decltype(src.l_receiptdate)::value_type);
    cudaMalloc((void**)&dst.l_receiptdate, columnSize);
    cudaMemcpy(dst.l_receiptdate, src.l_receiptdate.data(), columnSize, cudaMemcpyHostToDevice);

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

    lineitem_table_device_t lineitem;
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

    // char* l_returnflag, char* l_linestatus, int64_t* l_quantity, int64_t* l_extendedprice, int64_t* l_discount, int64_t* l_tax, uint32_t* l_shipdate
    query_1_kernel<<<numBlocks, blockSize>>>(N,
        lineitem.l_returnflag, lineitem.l_linestatus, lineitem.l_quantity, lineitem.l_extendedprice, lineitem.l_discount, lineitem.l_tax, lineitem.l_shipdate);
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
