#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <chrono>
#include <memory>

#include <cuda_runtime.h>
#include <sys/types.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <thread>

#include "utils.hpp"
#include "zipf.hpp"

#include "cuda_utils.cuh"
#include "cuda_allocator.hpp"
#include "numa_allocator.hpp"
#include "indexes.cuh"

using namespace std;

static const int blockSize = 128;
static const unsigned maxRepetitions = 1;
static const unsigned activeLanes = 32;
static const unsigned defaultNumLookups = 2048;// 1e8;
static unsigned defaultNumElements = 2048;// 1e7;

using index_key_t = uint32_t;
using value_t = uint32_t;
//using index_type = lower_bound_index<key_t, value_t>;
using index_type = harmonia_index<key_t, value_t>;

using indexed_allocator_t = cuda_allocator<key_t>;
using lookup_keys_allocator_t = cuda_allocator<key_t>;

template<class IndexStructureType>
__global__ void lookup_kernel(const IndexStructureType index_structure, unsigned n, const key_t* __restrict__ keys, value_t* __restrict__ tids) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int i = index;
    uint32_t active_lanes = __ballot_sync(FULL_MASK, i < n);
    while (active_lanes) {
        bool active = i < n;/*
        if (active) {
            printf("key[%d] = %d\n", i, keys[i]);
        }*/
//        auto tid = index_structure.lookup(keys[i]);
        auto tid = index_structure.cooperative_lookup(active, keys[i]);
        if (active) {
            tids[i] = tid;
//            printf("key[%d] = %d tids[%d] = %d\n", i, keys[i], i, tids[i]);
        }

        i += stride;
        active_lanes = __ballot_sync(FULL_MASK, i < n);
    }
}

template<class T>
__device__ T atomic_add_sat(T* address, T val, T saturation) {
    unsigned expected, update, old;
    old = *address;
    do {
        expected = old;
        update = (old + val > saturation) ? saturation : old + val;
        old = atomicCAS(address, expected, update);
    } while (expected != old);
    return old;
}



#if 0
template<
    unsigned BLOCK_THREADS,
    unsigned ITEMS_PER_THREAD,
    class    IndexStructureType>
__global__ void lookup_kernel_with_sorting(const IndexStructureType index_structure, unsigned n, const key_t* __restrict__ keys, value_t* __restrict__ tids) {
    enum { ITEMS_PER_ITERATION = BLOCK_THREADS*ITEMS_PER_THREAD };

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
    typedef cub::BlockLoad<key_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoad;

//    typedef cub::BlockRadixSort<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;
    typedef cub::BlockRadixSort<key_t, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;

    __shared__ union TempStorage {
        // Allocate shared memory for BlockLoad
        typename BlockLoad::TempStorage load;

        typename BlockRadixSortT::TempStorage sort;
    } temp_storage;

//    __shared__ uint32_t exhausted_warps;

    __shared__ key_t buffer[ITEMS_PER_ITERATION];
    __shared__ uint32_t buffer_pos;

    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;


    // initialize shared memory variables
    if (warp_id == 0 && lane_id == 0) {
//        exhausted_warps = 0;
        buffer_pos = 0;
    }
    __syncthreads(); // ensure that all shared variables are initialized


    uint32_t read = 0;

    const uint32_t iteration_count = (n + ITEMS_PER_ITERATION - 1) / ITEMS_PER_ITERATION;


    using key_array_t = key_t[ITEMS_PER_THREAD];
    key_t* thread_data_raw = &buffer[threadIdx.x*ITEMS_PER_THREAD];
    key_array_t& thread_data = reinterpret_cast<key_array_t&>(*thread_data_raw);


    for (int i = 0; i < iteration_count; ++i) {

        if (lane_id == 0) {
            buffer_pos = 0;
        }

        int valid_items = min(ITEMS_PER_ITERATION, n - read);
        // Load a segment of consecutive items that are blocked across threads
        BlockLoad(temp_storage.load).Load(keys + read, thread_data, valid_items);

        __syncthreads();




        // we only perform the sort step when the buffer is completely filled
        if (valid_items == ITEMS_PER_ITERATION) {
            BlockRadixSortT(temp_storage.sort).SortBlockedToStriped(thread_data, 0, 21); // TODO
             __syncthreads();
        }


        const auto old = atomic_add_sat(&buffer_pos, 32u, (unsigned)ITEMS_PER_ITERATION);
        const auto actual_count = min(ITEMS_PER_ITERATION - old, 32);

        printf("warp: %d lane: %d - actual_count: %u\n", warp_id, lane_id, actual_count);

        read += valid_items;
         __syncthreads();
    }


    int i = index;
    uint32_t active_lanes = __ballot_sync(FULL_MASK, i < n);
    while (active_lanes) {
        bool active = i < n;
        if (active) {
            printf("key[%d] = %d\n", i, keys[i]);
        }
//        auto tid = index_structure.lookup(keys[i]);
        auto tid = index_structure.cooperative_lookup(active, keys[i]);
        if (active) {
            tids[i] = tid;
            printf("key[%d] = %d tids[%d] = %d\n", i, keys[i], i, tids[i]);
        }

        i += stride;
        active_lanes = __ballot_sync(FULL_MASK, i < n);
    }
}
#endif




#if 1
template<
    unsigned BLOCK_THREADS,
    unsigned ITEMS_PER_THREAD,
    class    IndexStructureType>
__global__ void lookup_kernel_with_sorting(const IndexStructureType index_structure, unsigned n, const key_t* __restrict__ keys, value_t* __restrict__ tids) {
    enum { ITEMS_PER_ITERATION = BLOCK_THREADS*ITEMS_PER_THREAD };

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
    typedef cub::BlockLoad<key_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoad;

    typedef cub::BlockRadixSort<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;
//    typedef cub::BlockRadixSort<key_t, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;

    __shared__ union TempStorage {
        // Allocate shared memory for BlockLoad
        typename BlockLoad::TempStorage load;

        typename BlockRadixSortT::TempStorage sort;
    } temp_storage;

    __shared__ uint64_t buffer[ITEMS_PER_ITERATION];
    __shared__ uint32_t buffer_pos;

    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

/*
    // initialize shared memory variables
    if (warp_id == 0 && lane_id == 0) {
        buffer_pos = 0;
    }
    __syncthreads(); // ensure that all shared variables are initialized
*/

    uint32_t read = 0;

    const uint32_t iteration_count = (n + ITEMS_PER_ITERATION - 1) / ITEMS_PER_ITERATION;


    using key_value_array_t = uint64_t[ITEMS_PER_THREAD];
    uint64_t* thread_data_raw = &buffer[threadIdx.x*ITEMS_PER_THREAD];
    key_value_array_t& thread_data = reinterpret_cast<key_value_array_t&>(*thread_data_raw);

    key_t input_thread_data[ITEMS_PER_THREAD];


    for (int i = 0; i < iteration_count; ++i) {


        int valid_items = min(ITEMS_PER_ITERATION, n - read);
        // Load a segment of consecutive items that are blocked across threads
        BlockLoad(temp_storage.load).Load(keys + read, input_thread_data, valid_items);

        __syncthreads();

        // reset shared memory variables
        if (lane_id == 0) {
            buffer_pos = 0;
        }

        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            buffer[threadIdx.x*ITEMS_PER_THREAD + i] = static_cast<uint64_t>(input_thread_data[i]);
        }


        __syncthreads();

        // we only perform the sort step when the buffer is completely filled
        if (valid_items == ITEMS_PER_ITERATION) {
            BlockRadixSortT(temp_storage.sort).SortBlockedToStriped(thread_data, 0, 21); // TODO
             __syncthreads();
        }


        const auto old = atomic_add_sat(&buffer_pos, 32u, (unsigned)ITEMS_PER_ITERATION);
        const auto actual_count = min(ITEMS_PER_ITERATION - old, 32);

        printf("warp: %d lane: %d - actual_count: %u\n", warp_id, lane_id, actual_count);

        read += valid_items;
   //      __syncthreads();
    }

/*
    int i = index;
    uint32_t active_lanes = __ballot_sync(FULL_MASK, i < n);
    while (active_lanes) {
        bool active = i < n;
        if (active) {
            printf("key[%d] = %d\n", i, keys[i]);
        }
//        auto tid = index_structure.lookup(keys[i]);
        auto tid = index_structure.cooperative_lookup(active, keys[i]);
        if (active) {
            tids[i] = tid;
            printf("key[%d] = %d tids[%d] = %d\n", i, keys[i], i, tids[i]);
        }

        i += stride;
        active_lanes = __ballot_sync(FULL_MASK, i < n);
    }*/

}
#endif






template<class IndexStructureType>
void generate_datasets(std::vector<key_t>& keys, std::vector<key_t>& lookups) {
    auto rng = std::default_random_engine {};

    // create random keys
    std::generate(keys.begin(), keys.end(), rng);
    std::sort(keys.begin(), keys.end());

    std::uniform_int_distribution<> lookup_distrib(0, keys.size() - 1);
    std::generate(lookups.begin(), lookups.end(), [&]() { return keys[lookup_distrib(rng)]; });

//    std::cout << "keys: " << stringify(keys.begin(), keys.end()) << std::endl;
//    std::cout << "lookups: " << stringify(lookups.begin(), lookups.end()) << std::endl;
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

template<class T>
struct abstract_device_array {
    T* ptr_;
    size_t size_;

    abstract_device_array() : ptr_(nullptr), size_(0) {}

    abstract_device_array(T* ptr, size_t size) : ptr_(ptr), size_(size) {}

    abstract_device_array(const abstract_device_array&) = delete;

    T* data() { return ptr_; }
};

template<class T, class Allocator>
struct device_array : abstract_device_array<T> {
    Allocator allocator_;

    device_array(T* ptr, size_t size, Allocator allocator) : abstract_device_array<T>(ptr, size), allocator_(allocator) {}

    device_array(const device_array&) = delete;

    ~device_array() {
        allocator_.deallocate(this->ptr_, sizeof(T)*this->size_);
    }
};


template<class T>
struct device_array<T, void> : abstract_device_array<T> {
    device_array(T* ptr, size_t size) : abstract_device_array<T>(ptr, size) {}
};


template<class T>
struct device_array_wrapper {
    std::unique_ptr<abstract_device_array<T>> device_array_;

    template<class Allocator>
    device_array_wrapper(T* ptr, size_t size, Allocator allocator) {
        device_array_ = std::make_unique<device_array<T, Allocator>>(ptr, size, allocator);
    }

    device_array_wrapper(T* ptr, size_t size) {
        device_array_ = std::make_unique<device_array<T, void>>(ptr, size);
    }

    T* data() { return device_array_->data(); }
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
//        lookup_kernel<<<numBlocks, blockSize>>>(index_structure, num_lookup_keys, d_lookup_keys, d_tids);
        lookup_kernel_with_sorting<blockSize, 8, IndexStructureType><<<numBlocks, blockSize>>>(index_structure, num_lookup_keys, d_lookup_keys, d_tids);
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
    indexed.resize(num_elements);
    lookup_keys.resize(defaultNumLookups);
    generate_datasets<index_type>(indexed, lookup_keys);

    // create gpu accessible vectors
    indexed_allocator_t indexed_allocator;
    auto d_indexed = create_device_array_from(indexed, indexed_allocator);
    lookup_keys_allocator_t lookup_keys_allocator;
    auto d_lookup_keys = create_device_array_from(lookup_keys, lookup_keys_allocator);
    index_type index = build_index<index_type>(indexed, d_indexed.data());

/*
auto v = std::vector<int, cuda_allocator<int>>();
auto t1 = create_device_array_from(indexed, a);
auto tt = create_device_array_from(v, a);
*/

    std::unique_ptr<value_t[]> h_tids;
    if constexpr (activeLanes < 32) {

    } else {
        auto result = run_lookup_benchmark(index, d_lookup_keys.data(), lookup_keys.size());
        h_tids.swap(result);
    }

#if 0
    // validate results
    printf("validating results...\n");
    for (unsigned i = 0; i < lookup_keys.size(); ++i) {
        if (lookup_keys[i] != indexed[h_tids[i]]) {
            printf("lookup_keys[%u]: %u indexed[h_tids[%u]]: %u\n", i, lookup_keys[i], i, indexed[h_tids[i]]);
            throw;
        }
    }
#endif

    return 0;
}
