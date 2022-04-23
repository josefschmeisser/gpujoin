#pragma once

#include <gpu_radix_partition.h>

extern "C" __launch_bounds__(1024, 2) __global__
    void gpu_contiguous_prefix_sum_int32(PrefixSumArgs args);

extern "C" __launch_bounds__(1024, 2) __global__
    void gpu_chunked_radix_partition_int32_int32(RadixPartitionArgs args);

extern "C" __launch_bounds__(1024, 1) __global__
    void gpu_chunked_laswwc_radix_partition_int32_int32(
        RadixPartitionArgs args, uint32_t shared_mem_bytes);

extern "C" __launch_bounds__(1024, 1) __global__
    void gpu_chunked_sswwc_radix_partition_v2_int32_int32(
        RadixPartitionArgs args, uint32_t shared_mem_bytes);
