#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <sys/types.h>
#include <functional>
#include <iostream>
#include <numeric>
#include <chrono>
#include <memory>

#include "tpch/common.hpp"
#include "zipf.hpp"

#include "cuda_utils.cuh"
#include "cuda_allocator.hpp"
#include "numa_allocator.hpp"
#include "indexes.cuh"

using namespace std;

static const int blockSize = 128;
static const unsigned maxRepetitions = 1;
static const unsigned activeLanes = 32;
static const unsigned defaultNumLookups = 1e7;
static unsigned defaultNumElements = 1e7;

/*
using namespace btree;
using namespace btree::cuda;
*/

/*
using index_store_policy = vector_to_managed_array;
using index_store_policy = vector_to_managed_array;
*/

using index_key_t = uint32_t;
using value_t = uint32_t;
using index_type = lower_bound_index<key_t, value_t>;

using indexed_allocator_t = cuda_allocator<key_t>;
using lookup_keys_allocator_t = cuda_allocator<key_t>;

template<class IndexStructureType>
__global__ void lookup_kernel(const IndexStructureType index_structure, unsigned n, const key_t* __restrict__ keys, value_t* __restrict__ tids) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int i = index;
    uint32_t active_lanes = __ballot_sync(FULL_MASK, i < n);
    while (active_lanes) {
        bool active = i < n;
        auto tid = index_structure.lookup(keys[i]);
        if (active) {
            tids[i] = tid;
            printf("tids[%d] = %d\n", i, tids[i]);
        }

        i += stride;
        active_lanes = __ballot_sync(FULL_MASK, i < n);
    }
}

#if 0
template<class IndexStructureType>
__global__ void lookup_kernel_with_sorting(const IndexStructureType index_structure, unsigned n, const uint32_t* __restrict__ keys, uint32_t* __restrict__ tids) {
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
#endif

template<class IndexStructureType>
void generate_datasets(std::vector<key_t>& keys, std::vector<key_t>& lookups) {
/*
    std::vector<index_key_t> keys(numElements);
    std::iota(keys.begin(), keys.end(), 0);


    // shuffle lookup keys
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(keys), std::end(keys), rng);

    // generate lookup keys
    std::vector<index_key_t> lookupKeys(numAugmentedLookups);
    std::iota(lookupKeys.begin(), lookupKeys.end(), 0);
    std::shuffle(std::begin(lookupKeys), std::end(lookupKeys), rng);
    // TODO zipfian lookup patterns

    // copy lookup keys
    index_key_t* d_lookupKeys;
    cudaMalloc(&d_lookupKeys, numAugmentedLookups*sizeof(index_key_t));
    cudaMemcpy(d_lookupKeys, lookupKeys.data(), numAugmentedLookups*sizeof(index_key_t), cudaMemcpyHostToDevice);


*/
}

template<class IndexStructureType>
IndexStructureType build_index(const std::vector<key_t>& h_keys, key_t* d_keys) {
/*
    auto tree = btree::construct(keys, 0.9);
    for (unsigned i = 0; i < numElements; ++i) {
        //printf("lookup %d\n", i);
        btree::payload_t value;
        bool found = btree::lookup(tree, keys[i], value);
        if (!found) throw 0;
    }
*/
    IndexStructureType index;
    index.construct(h_keys, d_keys);
    return index;
}


struct abstract_device_array {};

template<class T, class Allocator>
struct device_array : abstract_device_array {
    T* ptr_;
    size_t size_;
    Allocator allocator_;

    device_array(T* ptr, size_t size, Allocator allocator) : ptr_(ptr), size_(size), allocator_(allocator) {}

    device_array(const device_array&) = delete;

    ~device_array() {
        allocator_.deallocate(ptr_, sizeof(T)*size_);
    }

    T* data() { return ptr_; }
};


template<class T>
struct device_array<T, void> : abstract_device_array {
    T* ptr_;
    size_t size_;

    device_array(T* ptr, size_t size) : ptr_(ptr), size_(size) {}

    device_array(const device_array&) = delete;

    T* data() { return ptr_; }
};


template<class T>
struct device_array_wrapper {
    std::unique_ptr<abstract_device_array> device_array_;

    template<class Allocator>
    device_array_wrapper(T* ptr, size_t size, Allocator allocator) {
        device_array_ = std::make_unique<device_array<T, Allocator>>(ptr, size, allocator);
    }

    device_array_wrapper(T* ptr, size_t size) {
        device_array_ = std::make_unique<device_array<T, void>>(ptr, size);
    }
};


template<class T, class OutputAllocator, class InputAllocator>
auto create_device_array_from(std::vector<T, InputAllocator>& vec, OutputAllocator& allocator) {
    printf("different types\n");
    if constexpr (std::is_same<OutputAllocator, numa_allocator<T>>::value) {
        T* ptr = allocator.allocate(vec.size()*sizeof(T));
        std::memcpy(ptr, vec.data(), vec.size()*sizeof(T));
        return device_array_wrapper(ptr, vec.size(), allocator);
    } else if (std::is_same<OutputAllocator, cuda_allocator<T, true>>::value) {
        return device_array_wrapper(vec.data(), vec.size());
    } else if (std::is_same<OutputAllocator, cuda_allocator<T, false>>::value) {
        T* ptr;
        cudaMalloc((void**)&ptr, vec.size()*sizeof(T));
        cudaMemcpy(ptr, vec.data(), vec.size()*sizeof(T), cudaMemcpyHostToDevice);
        return device_array_wrapper(ptr, vec.size(), allocator);
    }
    throw std::runtime_error("not available");
}


template<class T, class OutputAllocator>
auto create_device_array_from(std::vector<T, OutputAllocator>& vec, OutputAllocator& allocator) {
    printf("same type\n");
    if constexpr (std::is_same<OutputAllocator, numa_allocator<T>>::value) {
        if (allocator.node() == vec.get_allocator().node()) {
            return device_array_wrapper(vec.data(), vec.size());
        } else {
            T* ptr = allocator.allocate(vec.size()*sizeof(T));
            std::memcpy(ptr, vec.data(), vec.size()*sizeof(T));
            return device_array_wrapper(ptr, vec.size(), allocator);
        }
    } else if (std::is_same<OutputAllocator, cuda_allocator<T, false>>::value) {
        return device_array_wrapper(vec.data(), vec.size());
    }
    throw std::runtime_error("not available");
}


template<class IndexStructureType>
auto run_lookup_benchmark(IndexStructureType index_structure, const key_t* d_lookup_keys, unsigned num_lookup_keys) {
    int numBlocks = (num_lookup_keys + blockSize - 1) / blockSize;
    printf("numblocks: %d\n", numBlocks);

    // create result array
    value_t* d_tids;
    cudaMalloc(&d_tids, num_lookup_keys*sizeof(value_t));

    printf("executing kernel...\n");
    auto kernelStart = std::chrono::high_resolution_clock::now();
    for (unsigned rep = 0; rep < maxRepetitions; ++rep) {
        lookup_kernel<<<numBlocks, blockSize>>>(index_structure, num_lookup_keys, d_lookup_keys, d_tids);
        cudaDeviceSynchronize();
    }
    const auto kernelStop = std::chrono::high_resolution_clock::now();
    const auto kernelTime = chrono::duration_cast<chrono::microseconds>(kernelStop - kernelStart).count()/1000.;
    std::cout << "Kernel time: " << kernelTime << " ms\n";
    std::cout << "GPU MOps: " << (maxRepetitions*num_lookup_keys/1e6)/(kernelTime/1e3) << endl;

    // transfer results
    std::unique_ptr<value_t[]> h_tids(new value_t[num_lookup_keys]);
    cudaMemcpy(h_tids.get(), d_tids, num_lookup_keys*sizeof(value_t), cudaMemcpyDeviceToHost);

    cudaFree(d_tids);

    return std::move(h_tids);
}

#if 0
void run_lane_limited_lookup_benchmark() {
    // calculate a factory by which the number of lookups has to be scaled in order to be able to serve all threads
    int lookupFactor = 32/activeLanes + ((32%activeLanes > 0) ? 1 : 0);
    const int numAugmentedLookups = numLookups*lookupFactor;
    std::cout << "lookup scale factor: " << lookupFactor << " numAugmentedLookups: " << numAugmentedLookups << std::endl;


    const int threadCount = ((numAugmentedLookups + 31) & (-32));
    std::cout << "n: " << numAugmentedLookups << " threadCount: " << threadCount << std::endl;
    decltype(std::chrono::high_resolution_clock::now()) kernelStart;


    std::cout << "active lanes: " << activeLanes << std::endl;
    kernelStart = std::chrono::high_resolution_clock::now();
    for (unsigned rep = 0; rep < maxRepetitions; ++rep) {
        btree_bulk_lookup_serialized<activeLanes><<<numBlocks, blockSize>>>(d_tree, numAugmentedLookups, d_lookupKeys, d_tids);
        cudaDeviceSynchronize();
    }

    const auto kernelStop = std::chrono::high_resolution_clock::now();
    const auto kernelTime = chrono::duration_cast<chrono::microseconds>(kernelStop - kernelStart).count()/1000.;
    std::cout << "Kernel time: " << kernelTime << " ms\n";
    std::cout << "GPU MOps: " << (maxRepetitions*numLookups/1e6)/(kernelTime/1e3) << endl;
}
#endif

int main(int argc, char** argv) {
    auto num_elements = defaultNumElements;
    if (argc > 1) {
        std::string::size_type sz;
        num_elements = std::stod(argv[1], &sz);
    }
    std::cout << "index size: " << num_elements << std::endl;

    // generate datasets
    std::vector<key_t> indexed, lookup_keys;
    generate_datasets<index_type>(indexed, lookup_keys);

    // create gpu accessible vectors
    indexed_allocator_t indexed_allocator;
    auto d_indexed = create_device_array_from(indexed, indexed_allocator); // TODO
    auto d_lookup_keys = lookup_keys.data(); // TODO
    index_type index = build_index<index_type>(indexed, d_indexed.data());


auto v = std::vector<int, cuda_allocator<int>>();
auto t1 = create_device_array_from(indexed, a);
auto tt = create_device_array_from(v, a);

    std::unique_ptr<value_t[]> h_tids;
    if constexpr (activeLanes < 32) {

    } else {
        auto result = run_lookup_benchmark(index, d_lookup_keys, lookup_keys.size());
        h_tids.swap(result);
    }
/*
    // validate results
    printf("validating results...\n");

    for (unsigned i = 0; i < num_lookup_keys; ++i) {
        //printf("tid: %lu key[i]: %lu\n", reinterpret_cast<uint64_t>(h_tids[i]), lookupKeys[i]);
        if (reinterpret_cast<value_t>(h_tids[i]) != lookupKeys[i]) {
            printf("i: %u tid: %lu key[i]: %u\n", i, reinterpret_cast<uint64_t>(h_tids[i]), lookupKeys[i]);
            throw;
        }
    }
*/
    return 0;
}
