#pragma once

#include "cuda_utils.cuh"
#include "device_array.hpp"
#include "gpu_prefix_sum.hpp"

template<class T>
constexpr unsigned padding_length() {
    return GPU_CACHE_LINE_SIZE / sizeof(T);
}

struct partition_offsets {
    device_array_wrapper<unsigned long long> offsets;
    device_array_wrapper<unsigned long long> local_offsets;

    partition_offsets() = default;

    template<class Allocator>
    partition_offsets(uint32_t max_chunks, uint32_t radix_bits, Allocator& allocator) {
        const auto chunks = 1; // we only consider contiguous histograms (at least for now)
        const auto num_partitions = gpu_prefix_sum::fanout(radix_bits);
        offsets = create_device_array<unsigned long long>(num_partitions * chunks);
        local_offsets = create_device_array<unsigned long long>(num_partitions * max_chunks);
    }
};

template<class T>
struct partitioned_relation {
    device_array_wrapper<T> relation;
    device_array_wrapper<uint64_t> offsets;

    partitioned_relation() = default;

    template<class Allocator>
    partitioned_relation(size_t len, uint32_t max_chunks, uint32_t radix_bits, Allocator& allocator) {
        const auto chunks = 1; // we only consider contiguous histograms (at least for now)
        const auto padding_len = ::padding_length<T>();
        const auto num_partitions = gpu_prefix_sum::fanout(radix_bits);
        const auto relation_len = len + (num_partitions * chunks) * padding_len;
        //printf("relation_len: %lu\n", relation_len);

        // allocate device accessible arrays
        relation = create_device_array<T>(relation_len);
        offsets = create_device_array<uint64_t>(num_partitions * chunks);
    }

    unsigned padding_length() const {
        return ::padding_length<T>();
    }
};
