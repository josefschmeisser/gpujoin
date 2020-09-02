#include <bits/stdint-intn.h>
#include <bits/stdint-uintn.h>
#include <iostream>
#include <math.h>

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
};

// CUDA kernel to add elements of two arrays
__global__
void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}


__device__ void ht_insert(int32_t k)
{
}

__device__ group* createGroup() {
    group* ptr = (group*)malloc(sizeof(group));
    printf("Thread %d got pointer: %p\n", threadIdx.x, ptr);
    memset(ptr, 0, sizeof(group));
    return ptr;
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

__global__
void query_1_kernel(int n, char* l_returnflag, char* l_linestatus, int64_t* l_quantity, int64_t* l_extendedprice, int64_t* l_discount, int64_t* l_tax, uint32_t* l_shipdate)
{
    //constexpr auto threshold_date = to_julian_day(2, 9, 1998); // 1998-09-02
    uint32_t threshold_date = 2451059;

    __shared__ group* ht[16];
    for (int i = threadIdx.x; i < 16; i += blockDim.x) {
        ht[i] = nullptr;
    }
    __syncthreads();

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        if (l_shipdate[i] > threshold_date) continue;

        uint16_t h = static_cast<uint16_t>(l_returnflag[i]) << 8;
        h |= l_linestatus[i];
        h &= 0b0111;
        group* groupPtr = ht[h];
        if (groupPtr != nullptr) {
            if (groupPtr->l_returnflag != l_returnflag[i] || groupPtr->l_linestatus != l_linestatus[i]) {
                // TODO handle collisions
                __threadfence();
                asm("trap;");
            }
        } else {
            // create new group
            groupPtr = createGroup();
            groupPtr->l_returnflag = l_returnflag[i];
            groupPtr->l_linestatus = l_linestatus[i];
            ht[h] = groupPtr; // FIXME may already be set; use CAS
        }
/*
        auto current_l_extendedprice = l_extendedprice[i];
        auto current_l_discount = l_discount[i];*/
        auto current_l_quantity = l_quantity[i];

        atomicAdd((unsigned long long int*)&groupPtr->sum_qty, (unsigned long long int)current_l_quantity);
    }

  //  __sync
    printf("done\n");
}

__global__ void mallocTest()
{
    char* ptr = (char*)malloc(123);
    printf("Thread %d got pointer: %p\n", threadIdx.x, ptr);
    free(ptr);
}

int main(void)
{
    // Set a heap size of 128 megabytes. Note that this must
    // be done before any kernel is launched.
    cudaThreadSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
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

    return 0;
}
