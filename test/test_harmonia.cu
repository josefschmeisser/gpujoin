#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>
#include <numeric>

#include "harmonia.cuh"
#include "thirdparty/cub_test/test_util.h"

using namespace std;
using namespace harmonia;

void test_root_only() {
    std::cout << "run test_root_only: ";
    std::vector<uint32_t> keys(7);
    std::iota(keys.begin(), keys.end(), 0);

    //template<class Key, class Value, unsigned fanout>
    harmonia_tree<uint32_t, uint32_t, 8 + 1, std::numeric_limits<uint32_t>::max()> tree;
    tree.construct(keys);

    unsigned i = 0;
    for (auto k : keys) {
//        std::cout << "lookup: " << k << std::endl;
        auto tree_idx = tree.lookup(k);
//        std::cout << "tree_idx: " << tree_idx << std::endl;
        AssertEquals(tree_idx, i);
        ++i;
    }
    std::cout << "ok" << std::endl;
}

template<unsigned node_size = 8>
void test_harmonia_host_lookup(unsigned n) {
    std::cout << "run test_harmonia_host_lookup with n=" << n << ": ";
    std::vector<uint32_t> keys(n);
    std::iota(keys.begin(), keys.end(), 0);

    //template<class Key, class Value, unsigned fanout>
    harmonia_tree<uint32_t, uint32_t, node_size + 1, std::numeric_limits<uint32_t>::max()> tree;
    tree.construct(keys);

    unsigned i = 0;
    for (auto k : keys) {
//        std::cout << "=== lookup: " << k << std::endl;
        auto tree_idx = tree.lookup(k);
//        std::cout << "tree_idx: " << tree_idx << std::endl;
        AssertEquals(tree_idx, i);
        ++i;
    }
    std::cout << "ok" << std::endl;
}

using harmonia_type = harmonia::harmonia_tree<
    uint32_t,
    uint32_t,
    8 + 1,
    std::numeric_limits<uint32_t>::max()>;
using harmonia_handle = harmonia_type::device_handle_t;

__global__ void lookup_kernel(const harmonia_handle& __restrict__ tree, unsigned n, const uint32_t* __restrict__ keys, uint32_t* __restrict__ tids) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int i = index;
    uint32_t active_lanes = __ballot_sync(FULL_MASK, i < n);
    while (active_lanes) {
        bool active = i < n;
        auto tid = harmonia_type::lookup(active, tree, keys[i]);
        if (active) {
            tids[i] = tid;
            printf("tids[%d] = %d\n", i, tids[i]);
        }

        i += stride;
        active_lanes = __ballot_sync(FULL_MASK, i < n);
    }
}

void test_harmonia_cuda_lookup(unsigned n) {
    std::cout << "run test_harmonia_cuda_lookup with n=" << n << ": ";
    std::vector<uint32_t> keys(n);
    std::iota(keys.begin(), keys.end(), 0);

    //template<class Key, class Value, unsigned fanout>
    using harmonia_t = harmonia_tree<uint32_t, uint32_t, 8 + 1, std::numeric_limits<uint32_t>::max()>;
    harmonia_t tree;
    tree.construct(keys);

    harmonia_t::device_handle_t device_handle;
    tree.create_device_handle(device_handle);

    uint32_t* d_keys;
    cudaMalloc(&d_keys, sizeof(uint32_t)*keys.size());
    cudaMemcpy(d_keys, keys.data(), sizeof(uint32_t)*keys.size(), cudaMemcpyHostToDevice);

    uint32_t* d_values;
    cudaMalloc(&d_values, sizeof(uint32_t)*keys.size());

    lookup_kernel<<<1, 32>>>(device_handle, keys.size(), d_keys, d_values);
    cudaDeviceSynchronize();

    std::vector<uint32_t> h_values;
    h_values.reserve(n);
    cudaMemcpy(h_values.data(), d_values, sizeof(uint32_t)*n, cudaMemcpyDeviceToHost);
    for (unsigned i = 0; i < n; ++i) {
        AssertEquals(h_values[i], i);
    }
    std::cout << "ok" << std::endl;
}

int main(int argc, char** argv) {
    test_root_only();
    test_harmonia_host_lookup(15); // two tree levels
    test_harmonia_host_lookup(280); // three tree levels
    test_harmonia_host_lookup(2000); // test underfull nodes (nodes with one child and no separator)
    test_harmonia_cuda_lookup(7); // root only
    test_harmonia_cuda_lookup(15); // two tree levels
    test_harmonia_cuda_lookup(280); // three tree levels
    test_harmonia_cuda_lookup(2000); // test underfull nodes (nodes with one child and no separator)
    return 0;
}
