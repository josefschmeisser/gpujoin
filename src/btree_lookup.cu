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

using namespace std;

static constexpr unsigned maxRepetitions = 1;
static constexpr unsigned activeLanes = 32;
static constexpr unsigned numLookups = 1e7;
static unsigned numElements = 1e7;

using namespace index_structures;

using my_key_t = uint32_t;
using payload_t = uint64_t;
using btree_t = btree<my_key_t, payload_t, std::numeric_limits<payload_t>::max()>;

__global__ void btree_bulk_lookup(const btree_t::NodeBase* __restrict__ root, unsigned n, const my_key_t* __restrict__ keys, payload_t* __restrict__ tids) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    //n = 32;

/*
    __shared__ uint8_t rootRaw[NodeBase::pageSize];
    if (threadIdx.x == 0) {
        memcpy(rootRaw, tree, NodeBase::pageSize);
    }
    const NodeBase* root = reinterpret_cast<NodeBase*>(rootRaw);
    __syncthreads();
*/

    for (int i = index; i < n; i += stride) {
        tids[i] = btree_t::lookup(root, keys[i]);
        //tids[i] = btree_t::lookup_with_hints(root, keys[i]);
    }
}

__global__ void btree_bulk_cooperative_lookup(const btree_t::NodeBase* __restrict__ root, unsigned threadCount, unsigned n, const my_key_t* __restrict__ keys, payload_t* __restrict__ tids) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
//if (index < 32 || index > 63) return;
    for (int i = index; i < threadCount; i += stride) {
  //      printf("i: %d\n", i);
        const auto tid = btree_t::cooperative_lookup(i < n, root, keys[i]);
        if (i < n) tids[i] = tid;
    }
}

/*
__global__ void btree_bulk_lookup_serialized(const NodeBase* __restrict__ tree, unsigned n, const btree::my_key_t* __restrict__ keys, payload_t* __restrict__ tids) {
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
*/

template<unsigned MAX_ACTIVE>
__global__ void btree_bulk_lookup_serialized(const btree_t::NodeBase* __restrict__ tree, unsigned n, const my_key_t* __restrict__ keys, payload_t* __restrict__ tids) {
    //enum { MAX_ACTIVE = 16, RUNS = 32 / MAX_ACTIVE };

    const int lane_id = threadIdx.x % 32;
    const uint32_t lane_mask = 1<<lane_id;
    const uint32_t active_lanes = (1<<MAX_ACTIVE) - 1; // mask of active lanes

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        //printf("ative mask: %d\n", active_lanes);
        if (active_lanes & lane_mask) {
            tids[i] = btree_t::lookup(tree, keys[i]);
            //tids[i] = btree_t::lookup_with_hints(tree, keys[i]);
        }
    }
}


int main(int argc, char** argv) {
    if (argc > 1) {
        std::string::size_type sz;
        numElements = std::stod(argv[1], &sz);
    }
    std::cout << "index size: " << numElements << std::endl;

    std::vector<my_key_t> keys(numElements);
    std::iota(keys.begin(), keys.end(), 0);

    btree_t tree;
    tree.construct(keys, 0.9);
    for (unsigned i = 0; i < numElements; ++i) {
        //printf("lookup %d\n", i);
        payload_t value;
        bool found = tree.lookup(keys[i], value);
        if (!found) throw 0;
    }

    // shuffle keys
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(keys), std::end(keys), rng);

    // calculate a factory by which the number of lookups has to be scaled in order to be able to serve all threads
    int lookupFactor = 32/activeLanes + ((32%activeLanes > 0) ? 1 : 0);
    const int numAugmentedLookups = numLookups*lookupFactor;
    std::cout << "lookup scale factor: " << lookupFactor << " numAugmentedLookups: " << numAugmentedLookups << std::endl;

    // generate lookup keys
    std::vector<my_key_t> lookupKeys(numAugmentedLookups);
    std::iota(lookupKeys.begin(), lookupKeys.end(), 0);
    std::shuffle(std::begin(lookupKeys), std::end(lookupKeys), rng);
    // TODO zipfian lookup patterns

    // copy lookup keys
    my_key_t* d_lookupKeys;
    cudaMalloc(&d_lookupKeys, numAugmentedLookups*sizeof(my_key_t));
    cudaMemcpy(d_lookupKeys, lookupKeys.data(), numAugmentedLookups*sizeof(my_key_t), cudaMemcpyHostToDevice);
    payload_t* d_tids;
    cudaMalloc(&d_tids, numAugmentedLookups*sizeof(decltype(d_tids)));

    //btree::prefetchTree(tree, 0);
    auto d_tree = tree.copy_btree_to_gpu(tree.root);

    int blockSize = 32;
    int numBlocks = (numAugmentedLookups + blockSize - 1) / blockSize;
    /*
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    int numBlocks = 32*numSMs;*/
    printf("numblocks: %d\n", numBlocks);

    printf("executing kernel...\n");
    const int threadCount = ((numAugmentedLookups + 31) & (-32));
    std::cout << "n: " << numAugmentedLookups << " threadCount: " << threadCount << std::endl;
    decltype(std::chrono::high_resolution_clock::now()) kernelStart;
    if constexpr (activeLanes < 32) {
        std::cout << "active lanes: " << activeLanes << std::endl;
        kernelStart = std::chrono::high_resolution_clock::now();
        for (unsigned rep = 0; rep < maxRepetitions; ++rep) {
            btree_bulk_lookup_serialized<activeLanes><<<numBlocks, blockSize>>>(d_tree, numAugmentedLookups, d_lookupKeys, d_tids);
            cudaDeviceSynchronize();
        }
    } else {
        kernelStart = std::chrono::high_resolution_clock::now();
        for (unsigned rep = 0; rep < maxRepetitions; ++rep) {
            btree_bulk_lookup<<<numBlocks, blockSize>>>(d_tree, numAugmentedLookups, d_lookupKeys, d_tids);
            //btree_bulk_cooperative_lookup<<<numBlocks, blockSize>>>(d_tree, threadCount, numAugmentedLookups, d_lookupKeys, d_tids);
            cudaDeviceSynchronize();
        }
    }
    const auto kernelStop = std::chrono::high_resolution_clock::now();
    const auto kernelTime = chrono::duration_cast<chrono::microseconds>(kernelStop - kernelStart).count()/1000.;
    std::cout << "Kernel time: " << kernelTime << " ms\n";
    std::cout << "GPU MOps: " << (maxRepetitions*numLookups/1e6)/(kernelTime/1e3) << endl;

    // TODO: implement
    if constexpr (activeLanes == 32) {
        // validate results
        printf("validating results...\n");
        std::unique_ptr<payload_t[]> h_tids(new payload_t[numElements]);
        cudaMemcpy(h_tids.get(), d_tids, numElements*sizeof(decltype(d_tids)), cudaMemcpyDeviceToHost);
        for (unsigned i = 0; i < numElements; ++i) {
            //printf("tid: %lu key[i]: %lu\n", reinterpret_cast<uint64_t>(h_tids[i]), lookupKeys[i]);
            if (reinterpret_cast<uint64_t>(h_tids[i]) != lookupKeys[i]) {
                printf("i: %u tid: %lu key[i]: %u\n", i, reinterpret_cast<uint64_t>(h_tids[i]), lookupKeys[i]);
                throw;
            }
        }
    }

    return 0;
/*
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

    return 0;*/
}
