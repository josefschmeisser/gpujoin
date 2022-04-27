#pragma once

#ifdef GPU_CACHE_LINE_SIZE
#undef GPU_CACHE_LINE_SIZE
#endif
#include <gpu_radix_partition.h>

extern "C" __launch_bounds__(1024, 2) __global__
    void gpu_contiguous_prefix_sum_int32(PrefixSumArgs args);

extern "C" __launch_bounds__(1024, 2) __global__
    void gpu_contiguous_prefix_sum_int64(PrefixSumArgs args);

extern "C" __launch_bounds__(1024, 2) __global__
    void gpu_chunked_radix_partition_int32_int32(RadixPartitionArgs args);
/*
extern "C" __launch_bounds__(1024, 2) __global__
    void gpu_chunked_radix_partition_int32_int64(RadixPartitionArgs args);
*/
extern "C" __launch_bounds__(1024, 2) __global__
    void gpu_chunked_radix_partition_int64_int64(RadixPartitionArgs args);

extern "C" __launch_bounds__(1024, 1) __global__
    void gpu_chunked_laswwc_radix_partition_int32_int32(
        RadixPartitionArgs args, uint32_t shared_mem_bytes);
/*
extern "C" __launch_bounds__(1024, 1) __global__
    void gpu_chunked_laswwc_radix_partition_int32_int64(
        RadixPartitionArgs args, uint32_t shared_mem_bytes);
*/
extern "C" __launch_bounds__(1024, 1) __global__
    void gpu_chunked_laswwc_radix_partition_int64_int64(
        RadixPartitionArgs args, uint32_t shared_mem_bytes);

extern "C" __launch_bounds__(1024, 1) __global__
    void gpu_chunked_sswwc_radix_partition_v2_int32_int32(
        RadixPartitionArgs args, uint32_t shared_mem_bytes);
/*
extern "C" __launch_bounds__(1024, 1) __global__
    void gpu_chunked_sswwc_radix_partition_v2_int32_int64(
        RadixPartitionArgs args, uint32_t shared_mem_bytes);
*/
extern "C" __launch_bounds__(1024, 1) __global__
    void gpu_chunked_sswwc_radix_partition_v2_int64_int64(
        RadixPartitionArgs args, uint32_t shared_mem_bytes);

struct partitioned_consumer_assign_tasks_args {
    // Input
    uint32_t rel_length;
    uint32_t rel_padding_length;
    unsigned long long* rel_partition_offsets;
    uint32_t radix_bits;
    // Output
    uint32_t* task_assignment;
};

__global__ void partitioned_consumer_assign_tasks(partitioned_consumer_assign_tasks_args args);
