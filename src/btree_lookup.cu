#include "btree.cuh"

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <numeric>
#include <sys/types.h>
#include <chrono>

#include "zipf.hpp"

using namespace std;

static constexpr unsigned numElements = 1e7;

int main() {

    std::vector<btree::key_t> keys(numElements);
    std::iota(keys.begin(), keys.end(), 0);

//    auto tree = btree::construct_dense(1e6, 0.7);
    auto tree = btree::construct(keys, 0.7);
    for (unsigned i = 0; i < numElements; ++i) {
        //printf("lookup %d\n", i);
        btree::payload_t value;
        bool found = btree::lookup(tree, keys[i], value);
        if (!found) throw 0;
    }

    int blockSize = 32;
    int numBlocks = (numElements + blockSize - 1) / blockSize;
/*
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    int numBlocks = 32*numSMs;*/
    printf("numblocks: %d\n", numBlocks);

    // shuffle keys
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(keys), std::end(keys), rng);
    // TODO zipfian lookup patterns

    btree::key_t* lookupKeys;
    cudaMalloc(&lookupKeys, numElements*sizeof(key_t));
    // TODO shuffle keys/Zipfian lookup patterns
    cudaMemcpy(lookupKeys, keys.data(), numElements*sizeof(key_t), cudaMemcpyHostToDevice);
    btree::payload_t* tids;
    cudaMallocManaged(&tids, numElements*sizeof(decltype(tids)));

//    btree::prefetchTree(tree);

    auto start = std::chrono::high_resolution_clock::now();
    btree::cuda::btree_bulk_lookup<<<numBlocks, blockSize>>>(tree, numElements, lookupKeys, tids);
    cudaDeviceSynchronize();
    auto kernelStop = std::chrono::high_resolution_clock::now();
    auto kernelTime = chrono::duration_cast<chrono::microseconds>(kernelStop - start).count()/1000.;
    std::cout << "Kernel time: " << kernelTime << " ms\n";
    std::cout << "GPU MOps: " << (numElements/1e6)/(kernelTime/1e3) << endl;

/*
    for (unsigned i = 0; i < numElements; ++i) {
        printf("tid: %lu\n", reinterpret_cast<uint64_t>(tids[i]));
    }
*/

    start = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < numElements; ++i) {
        btree::payload_t value;
        bool found = btree::lookup(tree, keys[i], value);
        if (!found) throw 0;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto cpuTime = chrono::duration_cast<chrono::microseconds>(stop - start).count()/1000.;
    std::cout << "CPU time: " << cpuTime << " ms\n";
    std::cout << "CPU MOps: " << (numElements/1e6)/(cpuTime/1e3) << endl;

    return 0;
}
