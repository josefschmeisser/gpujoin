#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <chrono>
#include <memory>

#include <cub/util_debug.cuh>

#include <numa-gpu/sql-ops/include/gpu_radix_partition.h>
#include <numa-gpu/sql-ops/cudautils/gpu_common.cu>
#include <numa-gpu/sql-ops/cudautils/radix_partition.cu>

#include "cuda_utils.cuh"
#include "cuda_allocator.hpp"
#include "numa_allocator.hpp"
#include "mmap_allocator.hpp"
#include "indexes.cuh"
#include "device_array.hpp"

#include "index_lookup_config.cuh"
#include "index_lookup_common.cuh"


static const int block_size = 64;
static const int grid_size = 1;
static const uint32_t radix_bits = 22;
static const uint32_t ignore_bits = 8;

template<class T> using device_allocator_t = cuda_allocator<T, cuda_allocation_type::device>;
template<class T> using device_index_allocator_t = cuda_allocator<T, cuda_allocation_type::zero_copy>;

namespace gpu_prefix_sum {

// same as in partition.rs
uint32_t fanout(uint32_t radix_bits) {
    return (1 << radix_bits);
}

template<class G, class B>
size_t state_size(G grid_size, B block_size) {
    cudaDeviceProp device_properties;
    const auto ret = cudaGetDeviceProperties(&device_properties, 0); // FIXME
    CubDebugExit(ret);

    const auto warp_size = device_properties.warpSize;
    return ((grid_size * block_size) / warp_size + warp_size);
}

} // end namespace gpu_prefix_sum


struct partition_offsets {
    //void* local_offsets;
    device_array_wrapper<unsigned long long> local_offsets;

    template<class Allocator>
    partition_offsets(uint32_t max_chunks, uint32_t radix_bits, Allocator& allocator) {
        const auto num_partitions = gpu_prefix_sum::fanout(radix_bits);
        local_offsets = create_device_array<unsigned long long>(num_partitions * max_chunks);
    }
};

template<class T>
constexpr unsigned padding_length() {
    return GPU_CACHE_LINE_SIZE / sizeof(T);
}

template<class T>
struct partitioned_relation {
    device_array_wrapper<T> relation;
    device_array_wrapper<uint64_t> offsets;

    template<class Allocator>
    partitioned_relation(size_t len, uint32_t max_chunks, uint32_t radix_bits, Allocator& allocator) {
        const auto chunks = 1; // we only consider contiguous histograms (at least for now)
        const auto padding_len = ::padding_length<T>();
        const auto num_partitions = gpu_prefix_sum::fanout(radix_bits);
        const auto relation_len = len + (num_partitions * chunks) * padding_len;

        // allocate device accessible arrays
        relation = create_device_array<T>(relation_len);
        offsets = create_device_array<uint64_t>(num_partitions * chunks);
    }

    unsigned padding_length() const {
        return ::padding_length<T>();
    }
};

#if 0
const int N = 1 << 20;

__global__ void kernel(float *x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i));
    }
}

int main()
{
    const int num_streams = 8;

    cudaStream_t streams[num_streams];
    float *data[num_streams];

    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
 
        cudaMalloc(&data[i], N * sizeof(float));
        
        // launch one worker kernel per stream
        kernel<<<1, 64, 0, streams[i]>>>(data[i], N);

        // launch a dummy kernel on the default stream
        kernel<<<1, 1>>>(0, 0);
    }

    cudaDeviceReset();

    return 0;
}
#endif


__global__ void partition_kernel(float *x, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
    }
    /*
    template <typename K, typename V>
    __device__ void gpu_chunked_laswwc_radix_partition(RadixPartitionArgs &args, uint32_t shared_mem_bytes)*/

}

__global__ void join_kernel(float *x, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
    }
}

int main(int argc, char** argv) {
    auto num_elements = default_num_elements;
    auto num_lookups = default_num_lookups;
    if (argc > 1) {
        std::string::size_type sz;
        num_elements = std::stod(argv[1], &sz);
    }
    std::cout << "index size: " << num_elements << std::endl;

    // generate datasets
    std::vector<index_key_t, host_allocator_t<index_key_t>> indexed, lookup_keys;
    indexed.resize(num_elements);
    lookup_keys.resize(default_num_elements);
    generate_datasets<index_key_t, index_type>(dense, max_bits, indexed, lookup_keys);

    // create gpu accessible vectors
    indexed_allocator_t indexed_allocator;
    auto d_indexed = create_device_array_from(indexed, indexed_allocator);
    lookup_keys_allocator_t lookup_keys_allocator;
    auto d_lookup_keys = create_device_array_from(lookup_keys, lookup_keys_allocator);
    auto index = build_index<index_key_t, index_type>(indexed, d_indexed.data());

    // fetch device properties
    cudaDeviceProp device_properties;
    CubDebugExit(cudaGetDeviceProperties(&device_properties, 0));
    std::cout << "sharedMemPerBlock: " << device_properties.sharedMemPerBlock << std::endl;

    device_allocator_t<int> device_allocator;


/*
struct PrefixSumAndCopyWithPayloadArgs {
  // Inputs
  const void *const __restrict__ src_partition_attr;
  const void *const __restrict__ src_payload_attr;
  std::size_t const data_length;
  std::size_t const canonical_chunk_length;
  uint32_t const padding_length;
  uint32_t const radix_bits;
  uint32_t const ignore_bits;

  // State
  ScanState<unsigned long long> *const prefix_scan_state;
  unsigned long long *const __restrict__ tmp_partition_offsets;

  // Outputs
  void *const __restrict__ dst_partition_attr;
  void *const __restrict__ dst_payload_attr;
  unsigned long long *const __restrict__ partition_offsets;
};
*/

    // dummy payloads
    auto d_payloads = create_device_array<int32_t>(num_lookups);

    // allocate output arrays
    auto dst_partition_attr = create_device_array<index_key_t>(num_lookups);
    auto dst_payload_attrs = create_device_array<int32_t>(num_lookups);

    //ScanState<unsigned long long>* prefix_scan_state; // see: device_exclusive_prefix_sum_initialize
    const auto prefix_scan_state_len = gpu_prefix_sum::state_size(grid_size, block_size);
    auto prefix_scan_state = create_device_array<ScanState<unsigned long long>>(prefix_scan_state_len);

    partition_offsets offsets(grid_size, radix_bits, device_allocator);
    partitioned_relation<index_key_t> partitioned_relation_inst(num_lookups, grid_size, radix_bits, device_allocator);

    PrefixSumAndCopyWithPayloadArgs prefix_sum_and_copy_args {
        // Inputs
        d_lookup_keys.data(),
        d_payloads.data(),
        num_elements,
        0, // TODO check: not used?
        partitioned_relation_inst.padding_length(),
        radix_bits,
        ignore_bits,
        // State
        prefix_scan_state.data(),
        offsets.local_offsets.data(),
        // Outputs
        dst_partition_attr.data(),
        dst_payload_attrs.data(),
        offsets.local_offsets.data()
    };


    //__host__ ​cudaError_t cudaLaunchCooperativeKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream ) 
    /*
    template <typename K, typename V>
    __device__ void gpu_contiguous_prefix_sum_and_copy_with_payload(args)
    */
    cudaStream_t scan_stream;
    CubDebugExit(cudaStreamCreate(&scan_stream));

    //const void* func = &gpu_contiguous_prefix_sum_and_copy_with_payload<int, int>;

    void* args[1];
    args[0] = &prefix_sum_and_copy_args;
    CubDebugExit(cudaLaunchCooperativeKernel(
        //func,
        (void*)gpu_contiguous_prefix_sum_and_copy_with_payload_int32_int32,
        dim3(grid_size),
        dim3(block_size),
        args,
        device_properties.sharedMemPerBlock,
        scan_stream
    ));
    cudaDeviceSynchronize();
    printf("gpu_contiguous_prefix_sum_and_copy_with_payload_int32_int32 done\n");

/*
// Arguments to the partitioning function.
//
// Note that the struct's layout must be kept in sync with its counterpart in
// Rust.
struct RadixPartitionArgs {
  // Inputs
  const void *const __restrict__ join_attr_data;
  const void *const __restrict__ payload_attr_data;
  std::size_t const data_length;
  uint32_t const padding_length;
  uint32_t const radix_bits;
  uint32_t const ignore_bits;
  const unsigned long long *const __restrict__ partition_offsets;

  // State
  uint32_t *const __restrict__ tmp_partition_offsets;
  char *const __restrict__ l2_cache_buffers;
  char *const __restrict__ device_memory_buffers;
  uint64_t const device_memory_buffer_bytes;

  // Outputs
  void *const __restrict__ partitioned_relation;
};
*/

    RadixPartitionArgs radix_partition_args {
        // Inputs
        dst_partition_attr.data(),
        dst_payload_attrs.data(),
        num_lookups,
        partitioned_relation_inst.padding_length(),
        radix_bits,
        ignore_bits,
        offsets.local_offsets.data(),
        // State
        nullptr,
        nullptr,
        nullptr,
        0,
        // Outputs
        partitioned_relation_inst.relation.data()
    };

    /*
    template <typename K, typename V>
    __device__ void gpu_chunked_laswwc_radix_partition(RadixPartitionArgs &args, uint32_t shared_mem_bytes);
    */

    //gpu_chunked_laswwc_radix_partition<<<1, 64>>>(args, );

    gpu_chunked_laswwc_radix_partition_int32_int32<<<grid_size, block_size>>>(radix_partition_args, device_properties.sharedMemPerBlock);
    cudaDeviceSynchronize();
    printf("gpu_chunked_laswwc_radix_partition_int32_int32 done\n");

#if 0
    cudaStream_t partition_stream, join_stream;
    cudaStreamCreate(partition_stream);
    cudaStreamCreate(join_stream);


    partition_kernel<<<1, 64, 0, partition_stream>>>();
    join_kernel<<<1, 64, 0, join_stream>>>();
    cudaDeviceSynchronize();

#endif
    cudaDeviceReset();

    return 0;
}
