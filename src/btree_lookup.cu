#include "btree.hpp"

#include <bits/stdint-uintn.h>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <sys/types.h>
#include <chrono>

using namespace std;

static constexpr unsigned numElements = 1e8;
static constexpr btree::payload_t invalidTid = std::numeric_limits<btree::payload_t>::max();

namespace gpu {

using btree::key_t;
using btree::payload_t;
using btree::Node;

__device__ unsigned naive_lower_bound(Node* node, key_t key) {
    unsigned lower = 0;
    unsigned upper = node->count;
    do {
        unsigned mid = ((upper - lower) / 2) + lower;
        if (key < node->keys[mid]) {
            upper = mid;
        } else if (key > node->keys[mid]) {
            lower = mid + 1;
        } else {
            return mid;
        }
    } while (lower < upper);
    return lower;
}

__device__ payload_t btree_lookup(Node* tree, key_t key) {
    Node* node = tree;
    while (!node->isLeaf) {
        unsigned pos = naive_lower_bound(node, key);
        //printf("inner pos: %d\n", pos);
        node = reinterpret_cast<Node*>(node->payloads[pos]);
        if (node == nullptr) {
            return invalidTid;
        }
    }

    unsigned pos = naive_lower_bound(node, key);
    //printf("leaf pos: %d\n", pos);
    if ((pos < node->count) && (node->keys[pos] == key)) {
        return node->payloads[pos];
    }

    return invalidTid;
}

__device__ unsigned lower_bound_with_hint(Node* node, key_t key, float hint) {
    return 0;
}

__global__ void btree_bulk_lookup(Node* tree, unsigned n, uint32_t* keys, payload_t* tids) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    //printf("index: %d stride: %d\n", index, stride);

    for (int i = index; i < n; i += stride) {
        tids[i] = btree_lookup(tree, keys[i]);
    }
}

}

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

    btree::key_t* lookupKeys;
    cudaMalloc(&lookupKeys, numElements*sizeof(key_t));
    // TODO shuffle keys/Zipfian lookup patterns
    cudaMemcpy(lookupKeys, keys.data(), numElements*sizeof(key_t), cudaMemcpyHostToDevice);
    btree::payload_t* tids;
    cudaMallocManaged(&tids, numElements*sizeof(decltype(tids)));

    btree::prefetchTree(tree);

    auto start = std::chrono::high_resolution_clock::now();
    gpu::btree_bulk_lookup<<<numBlocks, blockSize>>>(tree, numElements, lookupKeys, tids);
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
