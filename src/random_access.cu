#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <sys/types.h>
#include <functional>
#include <iostream>
#include <numeric>
#include <chrono>
#include <memory>
#include <random>
#include <algorithm>

#include "allocators.hpp"
#include "device_array.hpp"
#include "utils.hpp"

using namespace std;

static constexpr unsigned repetitions = 10;
static constexpr unsigned blockSize = 512;
static unsigned numElements = 1e9;
static unsigned numAccesses = 1e9;

using lookup_t = uint32_t;

template<class T> using host_allocator_t = std::allocator<T>;
template<class T> using device_allocator_t = cuda_allocator<T, cuda_allocation_type::zero_copy>;

#if 0
__global__ void random_access(const lookup_t* __restrict__ src, const lookup_t* __restrict__ lookup, lookup_t* __restrict__ dst, unsigned n, unsigned rounds) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        lookup_t current = lookup[i];
        for (unsigned round = 0; round < rounds; ++round) {
            current = src[current];
        }
        dst[i] = current;
    }
}
#endif

__device__ uint32_t hash_key(uint32_t k) {
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k;
}

template<unsigned BLOCK_THREADS>
__global__ void random_access(const lookup_t* __restrict__ data, unsigned size, unsigned n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    __shared__ lookup_t results[BLOCK_THREADS];
    for (int i = index; i < n; i += stride) {
        auto pos = hash_key(i) & (size - 1);
        //auto pos = i; // sequential access
        lookup_t current = data[pos];
        (void)current;
        results[threadIdx.x] += current;
    }
}

int main(int argc, char** argv) {
    int numBlocks = (numAccesses + blockSize - 1) / blockSize;
    /*
    if (argc > 1) {
        blockSize = std::atoi(argv[1]);
    }
    printf("numblocks: %d\n", numBlocks);*/

    printf("numAccesses: %u\n", numAccesses);
    double transfered_gib = 1.*repetitions*numAccesses*sizeof(lookup_t)/std::pow(1024, 3.);
    std::cout << "to transfer: " << transfered_gib << " GiB" << std::endl;

    std::vector<lookup_t, host_allocator_t<lookup_t>> lookup_data(numElements);
    {
        // generate data
        std::iota(lookup_data.begin(), lookup_data.end(), 0);
        /*
        // shuffle keys
        auto rng = std::default_random_engine {};
        std::shuffle(std::begin(lookup_data), std::end(lookup_data), rng);
        */
    }

    // create gpu accessible vectors
    device_allocator_t<lookup_t> device_allocator;
    auto d_lookup_data = create_device_array_from(lookup_data, device_allocator);

    printf("executing kernel...\n");
    const auto kernelStart = std::chrono::high_resolution_clock::now();
    for (unsigned rep = 0; rep < repetitions; ++rep) {
        random_access<blockSize><<<numBlocks, blockSize>>>(d_lookup_data.data(), numElements, numAccesses);
        cudaDeviceSynchronize();
    }
    const auto kernelStop = std::chrono::high_resolution_clock::now();
    const double kernelTime = chrono::duration_cast<chrono::microseconds>(kernelStop - kernelStart).count()/1000.;
    std::cout << "Kernel time: " << kernelTime << " ms\n";
    std::cout << "GPU MOps: " << (repetitions*numAccesses/1e6)/(kernelTime/1e3) << endl;

    double rate = transfered_gib/(kernelTime/1000.);
    std::cout << "transfer rate: " << rate << " GiB/s" << std::endl;

    return 0;
}
