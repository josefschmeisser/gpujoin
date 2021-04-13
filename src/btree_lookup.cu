#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <numeric>
#include <chrono>
#include <memory>

#include "zipf.hpp"

#include "btree.cuh"
#include "btree.cu"

using namespace std;

static constexpr unsigned maxRepetitions = 10;
static unsigned numElements = 1e8;

using namespace btree;
using namespace btree::cuda;

__global__ void btree_bulk_lookup(const Node* __restrict__ tree, unsigned n, const btree::key_t* __restrict__ keys, payload_t* __restrict__ tids) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    __shared__ uint8_t rootRaw[Node::pageSize];
    if (threadIdx.x == 0) {
        memcpy(rootRaw, tree, Node::pageSize);
    }
    const Node* root = reinterpret_cast<Node*>(rootRaw);
    __syncthreads();

    for (int i = index; i < n; i += stride) {
        //tids[i] = btree_lookup(root, keys[i]);
        tids[i] = btree::cuda::btree_lookup_with_hints(root, keys[i]);
    }
}


__global__ void btree_bulk_lookup_serialized(const Node* __restrict__ tree, unsigned n, const btree::key_t* __restrict__ keys, payload_t* __restrict__ tids) {
    enum { MAX_ACTIVE = 16, RUNS = 32 / MAX_ACTIVE };

    const int lane_id = threadIdx.x % 32;
    const int lane_mask = 1<<lane_id;
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        // mask of active lanes
        uint32_t active_lanes = (1<<MAX_ACTIVE) - 1;
        for (unsigned j = 0; j < RUNS; ++j) {
            printf("ative mask: %d\n", active_lanes);
            if (active_lanes & lane_mask) {
                tids[i] = btree_lookup(tree, keys[i]);
                //tids[i] = btree::cuda::btree_lookup_with_hints(tree, keys[i]);
            }
            active_lanes <<= MAX_ACTIVE;
        }
    }
}


int main(int argc, char** argv) {
    if (argc > 1) {
        std::string::size_type sz;
        numElements = std::stod(argv[1], &sz);
    }
    std::cout << "index size: " << numElements << std::endl;

    std::vector<btree::key_t> keys(numElements);
    std::iota(keys.begin(), keys.end(), 0);

    auto tree = btree::construct(keys, 0.9);
    for (unsigned i = 0; i < numElements; ++i) {
        //printf("lookup %d\n", i);
        btree::payload_t value;
        bool found = btree::lookup(tree, keys[i], value);
        if (!found) throw 0;
    }

    int blockSize = 512;
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
    cudaMalloc(&lookupKeys, numElements*sizeof(btree::key_t));
    // TODO shuffle keys/Zipfian lookup patterns
    cudaMemcpy(lookupKeys, keys.data(), numElements*sizeof(btree::key_t), cudaMemcpyHostToDevice);
    btree::payload_t* d_tids;
    cudaMalloc(&d_tids, numElements*sizeof(decltype(d_tids)));

    //btree::prefetchTree(tree, 0);
    auto d_tree = tree;
    d_tree = btree::copy_btree_to_gpu(tree);

    printf("executing kernel...\n");
    const auto kernelStart = std::chrono::high_resolution_clock::now();
    for (unsigned rep = 0; rep < maxRepetitions; ++rep) {
        btree_bulk_lookup_serialized<<<numBlocks, blockSize>>>(d_tree, numElements, lookupKeys, d_tids);
        cudaDeviceSynchronize();
    }
    const auto kernelStop = std::chrono::high_resolution_clock::now();
    const auto kernelTime = chrono::duration_cast<chrono::microseconds>(kernelStop - kernelStart).count()/1000.;
    std::cout << "Kernel time: " << kernelTime << " ms\n";
    std::cout << "GPU MOps: " << (maxRepetitions*numElements/1e6)/(kernelTime/1e3) << endl;

    // validate results
    printf("validating results...\n");
    std::unique_ptr<btree::payload_t[]> h_tids(new btree::payload_t[numElements]);
    cudaMemcpy(h_tids.get(), d_tids, numElements*sizeof(decltype(d_tids)), cudaMemcpyDeviceToHost);
    for (unsigned i = 0; i < numElements; ++i) {
        //printf("tid: %lu key[i]: %lu\n", reinterpret_cast<uint64_t>(h_tids[i]), keys[i]);
        if (reinterpret_cast<uint64_t>(h_tids[i]) != keys[i]) {
            printf("i: %u tid: %lu key[i]: %u\n", i, reinterpret_cast<uint64_t>(h_tids[i]), keys[i]);
            throw;
        }
    }

    return 0;

    const auto cpuStart = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < numElements; ++i) {
        btree::payload_t value;
        bool found = btree::lookup(tree, keys[i], value);
        if (!found) throw 0;
    }
    const auto cpuStop = std::chrono::high_resolution_clock::now();
    const auto cpuTime = chrono::duration_cast<chrono::microseconds>(cpuStop - cpuStart).count()/1000.;
    std::cout << "CPU time: " << cpuTime << " ms\n";
    std::cout << "CPU MOps: " << (numElements/1e6)/(cpuTime/1e3) << endl;

    return 0;
}
