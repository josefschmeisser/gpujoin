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

#include "utils.hpp"

using namespace std;

static constexpr unsigned accessRounds = 1;
static constexpr unsigned maxRepetitions = 10;
static unsigned numElements = 1e9;

using lookup_t = uint32_t;
using placement_policy = vector_to_managed_array;

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

int main(int argc, char** argv) {
    double transfered_gib = accessRounds*numElements*sizeof(lookup_t)/std::pow(1024, 3.);
    std::cout << "total size: " << transfered_gib << " GiB" << std::endl;

    lookup_t* d_src, * d_lookup, * d_dst;
    {
        std::vector<lookup_t> keys(numElements);
        std::iota(keys.begin(), keys.end(), 0);

        // shuffle keys
        auto rng = std::default_random_engine {};
        std::shuffle(std::begin(keys), std::end(keys), rng);

        static placement_policy pp;
        d_src = pp(keys);

        vector_to_device_array f;
        d_lookup = f(keys);

        cudaMalloc(&d_dst, numElements*sizeof(lookup_t));
    }

    int blockSize = 512;
    int numBlocks = (numElements + blockSize - 1) / blockSize;

    printf("numblocks: %d\n", numBlocks);

    printf("executing kernel...\n");
    const auto kernelStart = std::chrono::high_resolution_clock::now();
    for (unsigned rep = 0; rep < maxRepetitions; ++rep) {
        random_access<<<numBlocks, blockSize>>>(d_src, d_lookup, d_dst, numElements, accessRounds);
        cudaDeviceSynchronize();
    }
    const auto kernelStop = std::chrono::high_resolution_clock::now();
    const auto kernelTime = chrono::duration_cast<chrono::microseconds>(kernelStop - kernelStart).count()/1000.;
    std::cout << "Kernel time: " << kernelTime << " ms\n";
    std::cout << "GPU MOps: " << (maxRepetitions*numElements/1e6)/(kernelTime/1e3) << endl;

    double rate = transfered_gib/(kernelTime/1000.);
    std::cout << "transfer rate: " << rate << " GiB/s" << std::endl;

    return 0;
}
