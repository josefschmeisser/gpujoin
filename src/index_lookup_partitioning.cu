#include "index_lookup_partitioning.cuh"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include <cub/util_debug.cuh>
#undef _Float16

#include <fast-interconnects/gpu_common.h>

#include "cuda_utils.cuh"
#include "cuda_allocator.hpp"
#include "numa_allocator.hpp"
#include "mmap_allocator.hpp"
#include "indexes.cuh"
#include "device_array.hpp"
#include "utils.hpp"
#include "measuring.hpp"
#include "device_properties.hpp"

#include "index_lookup_config.hpp"
#include "index_lookup_common.cuh"

#include "gpu_prefix_sum.hpp"
#include "gpu_radix_partition.cuh"
#include "partitioned_relation.hpp"

#ifdef NRDC
#include "src/gpu_radix_partition.cu"
#endif

using namespace measuring;

using dummy_payload_t = index_key_t; // the payload is not used
using rel_tuple_t = Tuple<index_key_t, dummy_payload_t>;

static const int parallel_streams = 2;
//static const int block_size = 128;
static int grid_size = 0;

// 48 kiB shared memory:
// laswwc max 8 bits
// sswwc v2 max 7 bits

struct PartitionedLookupArgs {
    // Input
    void* rel;
    uint32_t rel_length;
    uint32_t rel_padding_length;
    //uint64_t* rel_partition_offsets;
    unsigned long long* rel_partition_offsets;
    uint32_t* num_tasks;
    uint32_t* task_begin;
    uint32_t* task_end;
    uint32_t radix_bits;
    uint32_t ignore_bits;
    uint32_t block_size;
    // Output
    value_t* __restrict__ tids;
};

void dump_offsets(const partition_offsets& offsets) {
    auto h_offsets = offsets.offsets.to_host_accessible();
    std::cout << stringify(h_offsets.data(), h_offsets.data() + h_offsets.size()) << std::endl;
    auto h_local_offsets = offsets.local_offsets.to_host_accessible();
    std::cout << stringify(h_local_offsets.data(), h_local_offsets.data() + h_local_offsets.size()) << std::endl;
}

struct stream_state {
    cudaStream_t stream;

    uint32_t num_lookups;

    device_array_wrapper<dummy_payload_t> d_payloads;
    device_array_wrapper<value_t> d_dst_tids;
    device_array_wrapper<uint32_t> d_num_tasks;
    device_array_wrapper<uint32_t> d_task_begin;
    device_array_wrapper<uint32_t> d_task_end;

    device_array_wrapper<ScanState<unsigned long long>> d_prefix_scan_state;

    partition_offsets partition_offsets_inst;
    partitioned_relation<rel_tuple_t> partitioned_relation_inst;

    std::unique_ptr<PrefixSumArgs> prefix_sum_and_copy_args;
    std::unique_ptr<RadixPartitionArgs> radix_partition_args;
    std::unique_ptr<PartitionedLookupArgs> partitioned_lookup_args;
};

std::unique_ptr<stream_state> create_stream_state(const index_key_t* d_lookup_keys, uint32_t num_lookups, value_t* d_dst_tids) {
    const auto& config = get_experiment_config();
    const auto block_size = config.block_size;
    device_exclusive_allocator<int> device_allocator;
    auto state = std::make_unique<stream_state>();
    CubDebugExit(cudaStreamCreate(&state->stream));

    state->num_lookups = num_lookups;

    // initialize payloads
    {
        std::vector<dummy_payload_t> payloads;
        payloads.resize(num_lookups);
        std::iota(payloads.begin(), payloads.end(), 0);
        state->d_payloads = create_device_array_from(payloads, device_allocator);
    }

    // allocate output arrays
    uint32_t fanout = 1U << radix_bits;
    state->d_num_tasks = create_device_array<uint32_t>(1);
    state->d_task_begin = create_device_array<uint32_t>(fanout + grid_size);
    state->d_task_end = create_device_array<uint32_t>(fanout + grid_size);

    // see: device_exclusive_prefix_sum_initialize
    const auto prefix_scan_state_len = gpu_prefix_sum::state_size(grid_size, block_size);
    state->d_prefix_scan_state = create_device_array<ScanState<unsigned long long>>(prefix_scan_state_len);

    state->partition_offsets_inst = partition_offsets(grid_size, radix_bits, device_allocator);
    state->partitioned_relation_inst = partitioned_relation<rel_tuple_t>(num_lookups, grid_size, radix_bits, device_allocator);

    state->prefix_sum_and_copy_args = std::unique_ptr<PrefixSumArgs>(new PrefixSumArgs {
        // Inputs
        d_lookup_keys,
        num_lookups,
        0, // not used
        state->partitioned_relation_inst.padding_length(),
        radix_bits,
        config.partitioning_approach_ignore_bits,
        // State
        state->d_prefix_scan_state.data(),
        state->partition_offsets_inst.local_offsets.data(),
        // Outputs
        state->partition_offsets_inst.offsets.data()
    });

    state->radix_partition_args = std::unique_ptr<RadixPartitionArgs>(new RadixPartitionArgs {
        // Inputs
        d_lookup_keys,
        state->d_payloads.data(),
        num_lookups,
        state->partitioned_relation_inst.padding_length(),
        radix_bits,
        config.partitioning_approach_ignore_bits,
        state->partition_offsets_inst.local_offsets.data(),
        //state->partition_offsets_inst.offsets.data(),
        // State
        nullptr, // tmp_partition_offsets - used by gpu_chunked_sswwc_radix_partition_v2
        nullptr, // l2_cache_buffers - only used by gpu_chunked_sswwc_radix_partition_v2g
        nullptr, // device_memory_buffers - only used by gpu_chunked_hsswwc_* kernels
        0, // device_memory_buffer_bytes - only used by gpu_chunked_hsswwc_* kernels
        // Outputs
        state->partitioned_relation_inst.relation.data()
    });

    state->partitioned_lookup_args = std::unique_ptr<PartitionedLookupArgs>(new PartitionedLookupArgs {
        state->partitioned_relation_inst.relation.data(),
        static_cast<uint32_t>(state->partitioned_relation_inst.relation.size()), // TODO check
        state->partitioned_relation_inst.padding_length(),
        state->partition_offsets_inst.offsets.data(),
        state->d_num_tasks.data(),
        state->d_task_begin.data(),
        state->d_task_end.data(),
        radix_bits,
        config.partitioning_approach_ignore_bits,
        static_cast<uint32_t>(block_size),
        //state->d_dst_tids.data()
        d_dst_tids
    });

    return state;
}

template<class IndexStructureType>
__global__ void lookup_kernel(const IndexStructureType index_structure, device_size_t n, const rel_tuple_t* __restrict__ relation, value_t* __restrict__ tids) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int i = index;
    uint32_t active_lanes = __ballot_sync(FULL_MASK, i < n);
    while (active_lanes) {
        bool active = i < n;
        auto tid = index_structure.cooperative_lookup(active, relation[i].key);
        if (active) {
            tids[i] = tid;
        }

        i += stride;
        active_lanes = __ballot_sync(FULL_MASK, i < n);
    }
}

__global__ void partitioned_lookup_assign_tasks(PartitionedLookupArgs args) {
    const auto fanout = 1U << args.radix_bits;
    const auto max_tasks = gridDim.x + fanout;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const uint32_t rel_size = args.rel_length - args.rel_padding_length*fanout;
        const uint32_t avg_task_size = (rel_size + gridDim.x - 1U) / gridDim.x;

    uint32_t task_id = 0U;
    uint32_t partition_id = 0U;

    uint32_t partition_pos = args.rel_partition_offsets[partition_id];
    uint32_t partition_end = (partition_id + 1U < fanout)
            ? args.rel_partition_offsets[partition_id + 1U] - args.rel_padding_length
            : args.rel_length;

    while (task_id < max_tasks && partition_id < fanout) {
        args.task_begin[task_id] = partition_pos;

        if (partition_pos + avg_task_size < partition_end) {
        args.task_end[task_id] = partition_pos + avg_task_size;
        partition_pos += avg_task_size;
        }
        else {
        args.task_end[task_id] = partition_end;

        partition_id += 1U;
        partition_pos = args.rel_partition_offsets[partition_id];
        partition_end = (partition_id + 1U < fanout)
                    ? args.rel_partition_offsets[partition_id + 1U] - args.rel_padding_length
            : args.rel_length;
        }

        task_id += 1U;
    }

    *args.num_tasks = task_id;

        for (uint32_t i = task_id; i < max_tasks; ++i) {
            args.task_begin[i] = 0;
            args.task_end[i] = 0;
        }
    }
}

template<class TupleType, class IndexStructureType>
__global__ void partitioned_lookup_kernel_old(const IndexStructureType index_structure, const PartitionedLookupArgs args) {
    for (uint32_t task_id = blockIdx.x; task_id < *args.num_tasks; task_id += gridDim.x) {
        const TupleType* __restrict__ rel_begin = reinterpret_cast<const TupleType*>(args.rel) + args.task_begin[task_id];
        const TupleType* __restrict__ rel_end = reinterpret_cast<const TupleType*>(args.rel) + args.task_end[task_id];
        const uint32_t rel_size = rel_end - rel_begin;
        const uint32_t loop_limit = (rel_size + warpSize - 1) & ~(warpSize - 1); // round to next multiple of warpSize

        // cooperative lookup implementation
        for (uint32_t i = threadIdx.x; i < loop_limit; i += blockDim.x) {
            const bool active = i < rel_size;
            const TupleType tuple = active ? rel_begin[i] : TupleType();
            const auto tid = index_structure.cooperative_lookup(active, tuple.key);
            if (active) {
                args.tids[tuple.value] = tid;
            }
        }
    }
}

// TODO move
__device__ size_t partitioned_lookup_result_count = 0;

template<class T>
__forceinline__ __device__ void do_flush(T* dest, T* src, uint32_t count, uint32_t thread_rank) {
    for (uint32_t i = thread_rank; i < count; i += warpSize) {
        dest[i] = src[i];
    }
}

template<class TupleType, class IndexStructureType>
__global__ void partitioned_lookup_kernel(const IndexStructureType index_structure, const PartitionedLookupArgs args) {
    static constexpr uint32_t size_per_warp_soft = 128; // TODO tune
    static constexpr uint32_t size_per_warp_hard = size_per_warp_soft + 32;

    const auto lane_id = threadIdx.x & (warpSize - 1);
    const auto warp_id = threadIdx.x / warpSize;
    const int warp_count = args.block_size / warpSize;

    using key_type = decltype(TupleType::key);
    extern __shared__ uint8_t shared_mem[];
    size_t offset = 0;
    uint32_t* buffer_offset = reinterpret_cast<uint32_t*>(&shared_mem[offset]);
    offset += sizeof(uint32_t) * warp_count;
    key_type* buffer = reinterpret_cast<key_type*>(&shared_mem[offset]);
    //offset += sizeof(key_type) * warp_count * size_per_warp_hard;

    if (lane_id == 0) {
        buffer_offset[warp_id] = 0;
    }

    for (uint32_t task_id = blockIdx.x; task_id < *args.num_tasks; task_id += gridDim.x) {
        const TupleType* __restrict__ rel_begin = reinterpret_cast<const TupleType*>(args.rel) + args.task_begin[task_id];
        const TupleType* __restrict__ rel_end = reinterpret_cast<const TupleType*>(args.rel) + args.task_end[task_id];
        const uint32_t rel_size = rel_end - rel_begin;
        const uint32_t loop_limit = (rel_size + warpSize - 1) & ~(warpSize - 1); // round to next multiple of warpSize

        // cooperative lookup implementation
        for (uint32_t i = threadIdx.x; i < loop_limit; i += blockDim.x) {
            bool active = i < rel_size;
            const TupleType tuple = active ? rel_begin[i] : TupleType();
            const auto tid = index_structure.cooperative_lookup(active, tuple.key);

            static constexpr auto invalid_tid = std::numeric_limits<decltype(tid)>::max();
            active = active && tid != invalid_tid;

            const bool any_active = __ballot_sync(FULL_MASK, active);
            if (!any_active) continue;

            uint32_t pos = 0;
            if (active) {
                pos = tmpl_atomic_add(&buffer_offset[warp_id], 1U);
                buffer[warp_id * size_per_warp_hard + pos] = tuple.key;
            }

            const bool flush = __ballot_sync(FULL_MASK, pos >= size_per_warp_soft);
            if (!flush) continue;

            size_t global_pos;
            if (lane_id == 0) {
                global_pos = tmpl_atomic_add(&partitioned_lookup_result_count, static_cast<size_t>(buffer_offset[warp_id]));
            }
            global_pos = __shfl_sync(FULL_MASK, global_pos, 0);
            do_flush(args.tids + global_pos, &buffer[warp_id * size_per_warp_hard], buffer_offset[warp_id], lane_id);
            if (lane_id == 0) {
                buffer_offset[warp_id] = 0;
            }
        }
    }

    __syncwarp();
    size_t global_pos;
    if (lane_id == 0) {
        global_pos = tmpl_atomic_add(&partitioned_lookup_result_count, static_cast<size_t>(buffer_offset[warp_id]));
    }
    global_pos = __shfl_sync(FULL_MASK, global_pos, 0);
    do_flush(args.tids + global_pos, &buffer[warp_id * size_per_warp_hard], buffer_offset[warp_id], lane_id);
}

template<class TupleType, class IndexStructureType>
__global__ void partitioned_lookup_kernel_2(const IndexStructureType index_structure, const PartitionedLookupArgs args) {
    static constexpr uint32_t size_per_warp = 128; // TODO tune

    const auto lane_id = threadIdx.x & (warpSize - 1);
    const auto warp_id = threadIdx.x / warpSize;
    const int warp_count = args.block_size / warpSize;

    using key_type = decltype(TupleType::key);
    extern __shared__ uint8_t shared_mem[];
    size_t offset = 0;
    uint32_t* buffer_offset = reinterpret_cast<uint32_t*>(&shared_mem[offset]);
    offset += sizeof(uint32_t) * warp_count;
    key_type* buffer = reinterpret_cast<key_type*>(&shared_mem[offset]);

    if (lane_id == 0) {
        buffer_offset[warp_id] = 0;
    }

    for (uint32_t task_id = blockIdx.x; task_id < *args.num_tasks; task_id += gridDim.x) {
        const TupleType* __restrict__ rel_begin = reinterpret_cast<const TupleType*>(args.rel) + args.task_begin[task_id];
        const TupleType* __restrict__ rel_end = reinterpret_cast<const TupleType*>(args.rel) + args.task_end[task_id];
        const uint32_t rel_size = rel_end - rel_begin;
        const uint32_t loop_limit = (rel_size + warpSize - 1) & ~(warpSize - 1); // round to next multiple of warpSize

        // cooperative lookup implementation
        for (uint32_t i = threadIdx.x; i < loop_limit; i += blockDim.x) {
            bool active = i < rel_size;
            const TupleType tuple = active ? rel_begin[i] : TupleType();
            const auto tid = index_structure.cooperative_lookup(active, tuple.key);

            static constexpr auto invalid_tid = std::numeric_limits<decltype(tid)>::max();
            active = active && tid != invalid_tid;

            while (__ballot_sync(FULL_MASK, active)) {
                uint32_t pos = 0;
                if (active) {
                    pos = tmpl_atomic_add(&buffer_offset[warp_id], 1U);
                    if (pos < size_per_warp) {
                        buffer[warp_id * size_per_warp + pos] = tuple.key;
                        active = false;
                    }
                }

                const bool flush = __ballot_sync(FULL_MASK, pos >= size_per_warp);
                if (flush) {
                    size_t global_pos;
                    if (lane_id == 0) {
                        global_pos = tmpl_atomic_add(&partitioned_lookup_result_count, static_cast<size_t>(size_per_warp));
                    }
                    global_pos = __shfl_sync(FULL_MASK, global_pos, 0);
                    do_flush(args.tids + global_pos, &buffer[warp_id * size_per_warp], size_per_warp, lane_id);
                    if (lane_id == 0) {
                        buffer_offset[warp_id] = 0;
                    }
                }
            }
        }
    }

    __syncwarp();
    size_t global_pos;
    if (lane_id == 0) {
        global_pos = tmpl_atomic_add(&partitioned_lookup_result_count, static_cast<size_t>(buffer_offset[warp_id]));
    }
    global_pos = __shfl_sync(FULL_MASK, global_pos, 0);
    do_flush(args.tids + global_pos, &buffer[warp_id * size_per_warp], buffer_offset[warp_id], lane_id);
}

template<class K, class V>
std::string tmpl_to_string(const Tuple<K, V>& tuple) {
    return std::to_string(tuple.key);
}

void dump_partitions(const stream_state& state) {
    const auto offsets = state.partition_offsets_inst.offsets.to_host_accessible();
    const auto relation = state.partitioned_relation_inst.relation.to_host_accessible();
    const auto padding_length = state.radix_partition_args->padding_length;
    const auto fanout = 1U << state.radix_partition_args->radix_bits;

    for (size_t p = 0; p < offsets.size(); ++p) {
        std::cout << "partition " << p << " offset: " << offsets.data()[p] << std::endl;

        const uint32_t upper = (p + 1U < fanout) ? offsets.data()[p + 1U] - padding_length : relation.size();

        std::cout << "upper: " << upper << std::endl;

        for (size_t i = offsets.data()[p]; i < upper; ++i) {
            std::cout << relation.data()[i].key << ", ";
        }
        std::cout << std::endl;
    }
}

void dump_task_assignments(const stream_state& state) {
    const auto num_tasks = state.d_num_tasks.to_host_accessible();
    const auto task_begin = state.d_task_begin.to_host_accessible();
    const auto task_end = state.d_task_end.to_host_accessible();
    std::cout << "num tasks: " << *num_tasks.data() << std::endl;
    std::cout << "task begin: " << stringify(task_begin.data(), task_begin.data() + task_begin.size()) << std::endl;
    std::cout << "task end: " << stringify(task_end.data(), task_end.data() + task_end.size()) << std::endl;
}

template<class IndexedVectorType, class ResultVectorType>
bool validate_results(const std::vector<index_key_t>& lookup_keys, const IndexedVectorType& indexed, const ResultVectorType& tids) {
    const auto h_tids = tids.to_host_accessible();

    //std::cout << "tids: " << stringify(h_tids.data(), h_tids.data() + h_tids.size()) << std::endl;

    bool valid = true;
    for (size_t i = 0; i < lookup_keys.size(); ++i) {
        if (indexed[h_tids.data()[i]] != lookup_keys[i]) {
            valid = false;
            std::cerr << "missmatch at: " << i << std::endl;
        }
    }
    std::cout << "validation done" << std::endl;
    return valid;
}

template<class IndexStructureType>
void run_on_stream(stream_state& state, IndexStructureType& index_structure, const cudaDeviceProp& device_properties) {
    const auto& config = get_experiment_config();
    const auto block_size = config.block_size;

    // calculate prefix sum kernel shared memory requirement
    const auto required_shared_mem_bytes = ((block_size + (block_size >> LOG2_NUM_BANKS)) + gpu_prefix_sum::fanout(radix_bits)) * sizeof(uint64_t);
#ifdef DEBUG_INTERMEDIATE_STATE
    printf("required_shared_mem_bytes %lu\n", required_shared_mem_bytes);
#endif
    assert(required_shared_mem_bytes <= device_properties.sharedMemPerBlock);

    // prepare kernel arguments
    void* args[1];
    args[0] = state.prefix_sum_and_copy_args.get();

    //if constexpr (sizeof(index_key_t) == 4) {
    // 32 bit version
    execute_if<sizeof(index_key_t) == 4>::execute([&]() {
        //printf("execute: gpu_contiguous_prefix_sum_int32\n");
        // calculate prefix sum
        CubDebugExit(cudaLaunchCooperativeKernel(
            (void*)gpu_contiguous_prefix_sum_int32,
            dim3(grid_size),
            dim3(block_size),
            args,
            required_shared_mem_bytes,
            state.stream
        ));
    });
    // 64 bit version
    execute_if<sizeof(index_key_t) == 8>::execute([&]() {
        //printf("execute: gpu_contiguous_prefix_sum_int64\n");
        // calculate prefix sum
        CubDebugExit(cudaLaunchCooperativeKernel(
            (void*)gpu_contiguous_prefix_sum_int64,
            dim3(grid_size),
            dim3(block_size),
            args,
            required_shared_mem_bytes,
            state.stream
        ));
    });

#ifdef DEBUG_INTERMEDIATE_STATE
    cudaDeviceSynchronize();
    auto r = state.partition_offsets_inst.offsets.to_host_accessible();
    std::cout << "offsets: " << stringify(r.data(), r.data() + state.partition_offsets_inst.local_offsets.size()) << std::endl;
#endif

    // calculate radix partition kernel shared memory requirement
    // 32 bit version
    execute_if<sizeof(index_key_t) == 4>::execute([&]() {
        //gpu_chunked_radix_partition_int32_int32<<<grid_size, block_size, device_properties.sharedMemPerBlock, state.stream>>>(*state.radix_partition_args);
        gpu_chunked_laswwc_radix_partition_int32_int32<<<grid_size, block_size, device_properties.sharedMemPerBlock, state.stream>>>(*state.radix_partition_args, device_properties.sharedMemPerBlock);
        //gpu_chunked_sswwc_radix_partition_v2_int32_int32<<<grid_size, block_size, device_properties.sharedMemPerBlock, state.stream>>>(*state.radix_partition_args, device_properties.sharedMemPerBlock);
    });
    // 64 bit version
    execute_if<sizeof(index_key_t) == 8>::execute([&]() {
        //gpu_chunked_radix_partition_int64_int64<<<grid_size, block_size, device_properties.sharedMemPerBlock, state.stream>>>(*state.radix_partition_args);
        gpu_chunked_laswwc_radix_partition_int64_int64<<<grid_size, block_size, device_properties.sharedMemPerBlock, state.stream>>>(*state.radix_partition_args, device_properties.sharedMemPerBlock);
        //gpu_chunked_sswwc_radix_partition_v2_int64_int64<<<grid_size, block_size, device_properties.sharedMemPerBlock, state.stream>>>(*state.radix_partition_args, device_properties.sharedMemPerBlock);
    });

#ifdef DEBUG_INTERMEDIATE_STATE
    cudaDeviceSynchronize();
    auto r2 = state.partitioned_relation_inst.relation.to_host_accessible();
    std::cout << "result: " << stringify(r2.data(), r2.data() + state.partitioned_relation_inst.relation.size()) << std::endl;
    dump_partitions(state);
#endif

    partitioned_lookup_assign_tasks<<<grid_size, 1, 0, state.stream>>>(*state.partitioned_lookup_args);
#ifdef DEBUG_INTERMEDIATE_STATE
    cudaDeviceSynchronize();
    dump_task_assignments(state);
#endif

    size_t init = 0;
    cudaMemcpyToSymbol(partitioned_lookup_result_count, &init, sizeof(size_t));
    partitioned_lookup_kernel<rel_tuple_t><<<grid_size, block_size, device_properties.sharedMemPerBlock, state.stream>>>(index_structure.device_index, *state.partitioned_lookup_args);
}

template<class IndexType>
struct partitioning_approach<IndexType>::impl {
    std::vector<std::unique_ptr<stream_state>> stream_states;
};

template<class IndexType>
partitioning_approach<IndexType>::partitioning_approach()
    : _p_impl{std::make_unique<impl>()}
{}

template partitioning_approach<btree_type>::partitioning_approach();
template partitioning_approach<harmonia_type>::partitioning_approach();
template partitioning_approach<binary_search_type>::partitioning_approach();
template partitioning_approach<radix_spline_type>::partitioning_approach();
template partitioning_approach<no_op_type>::partitioning_approach();

template<class IndexType>
partitioning_approach<IndexType>::~partitioning_approach() = default;

template partitioning_approach<btree_type>::~partitioning_approach();
template partitioning_approach<harmonia_type>::~partitioning_approach();
template partitioning_approach<binary_search_type>::~partitioning_approach();
template partitioning_approach<radix_spline_type>::~partitioning_approach();
template partitioning_approach<no_op_type>::~partitioning_approach();

/*
template<class IndexType>
void partitioning_approach<IndexType>::initialize(query_data& d) {
    const auto& config = get_experiment_config();
    const auto& device_properties = get_device_properties(0);

    if (grid_size == 0) {
        grid_size = device_properties.multiProcessorCount;
    }

    if (config.num_lookups >= std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("config.num_lookups >= std::numeric_limits<uint32_t>::max()");
    }
    size_t remaining = config.num_lookups;
    size_t max_stream_portion = (config.num_lookups + parallel_streams) / parallel_streams;
    //printf("ALIGN_BYTES: %u\n", ALIGN_BYTES);
    max_stream_portion = (max_stream_portion + ALIGN_BYTES - 1) & -ALIGN_BYTES;
    const index_key_t* d_stream_lookup_keys = d.d_lookup_keys.data();
    value_t* d_stream_tids = d.d_tids.data();

    // create streams
    for (unsigned i = 0; i < parallel_streams; ++i) {
        size_t stream_portion = std::min(remaining, max_stream_portion);
        remaining -= stream_portion;
        printf("stream portion: %lu\n", stream_portion);
        auto state = create_stream_state(d_stream_lookup_keys, stream_portion, d_stream_tids);
        _p_impl->stream_states.push_back(std::move(state));

        d_stream_lookup_keys += stream_portion;
        d_stream_tids += stream_portion;
    }
}
*/

template<class IndexType>
void partitioning_approach<IndexType>::initialize(query_data& d) {
    const auto& config = get_experiment_config();
    const auto& device_properties = get_device_properties(0);

    if (grid_size == 0) {
        grid_size = device_properties.multiProcessorCount;
    }

    if (config.num_lookups >= std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("config.num_lookups >= std::numeric_limits<uint32_t>::max()");
    }

    int64_t window_size = config.partitioning_approach_window_size;
    window_size = window_size > 0 ? window_size : std::numeric_limits<decltype(window_size)>::max();
    //printf("window_size: %lu; config.num_lookups: %lu\n", window_size, config.num_lookups);
    constexpr size_t align_count = ALIGN_BYTES / sizeof(index_key_t);
    constexpr size_t align_mask = ~(align_count - 1ul);
    //printf("align_count: %lu; config.align_mask: %lu\n", align_count, align_mask);
    size_t max_stream_portion = std::min<size_t>(window_size, config.num_lookups);
    max_stream_portion = (max_stream_portion + parallel_streams) / parallel_streams;
    max_stream_portion = (max_stream_portion + align_count - 1ul) & align_mask;
    const index_key_t* d_stream_lookup_keys = d.d_lookup_keys.data();
    value_t* d_stream_tids = d.d_tids.data();
    //printf("max_stream_portion: %lu\n", max_stream_portion);

    // create streams
    size_t remaining = config.num_lookups;
    while (remaining > 0) {
        size_t stream_portion = std::min(remaining, max_stream_portion);
        remaining -= stream_portion;
        printf("stream portion: %lu\n", stream_portion);
        auto state = create_stream_state(d_stream_lookup_keys, stream_portion, d_stream_tids);
        _p_impl->stream_states.push_back(std::move(state));

        d_stream_lookup_keys += stream_portion;
        d_stream_tids += stream_portion;
    }
}

template void partitioning_approach<btree_type>::initialize(query_data& d);
template void partitioning_approach<harmonia_type>::initialize(query_data& d);
template void partitioning_approach<binary_search_type>::initialize(query_data& d);
template void partitioning_approach<radix_spline_type>::initialize(query_data& d);
template void partitioning_approach<no_op_type>::initialize(query_data& d);

/*
template<class IndexType>
void partitioning_approach<IndexType>::run(query_data& d, measurement& m) {
    const auto& device_properties = get_device_properties(0);

    IndexType& index_structure = *static_cast<IndexType*>(d.index_structure.get());
    for (const auto& state : _p_impl->stream_states) {
        run_on_stream(*state, index_structure, device_properties);
    }
    cudaDeviceSynchronize();
}
*/

template<class IndexType>
void partitioning_approach<IndexType>::run(query_data& d, measurement& m) {
    const auto& device_properties = get_device_properties(0);

    IndexType& index_structure = *static_cast<IndexType*>(d.index_structure.get());
    const size_t window_count = _p_impl->stream_states.size();
    for (size_t i = 0; i < window_count;) {
        const size_t stream_count = std::min<size_t>(parallel_streams, window_count - i);
        size_t j = 0;
        for (; j < stream_count; ++j) {
            const auto& state = _p_impl->stream_states[i + j];
            run_on_stream(*state, index_structure, device_properties);
        }
        i += j;
        cudaDeviceSynchronize();
    }
}

template void partitioning_approach<btree_type>::run(query_data& d, measurement& m);
template void partitioning_approach<harmonia_type>::run(query_data& d, measurement& m);
template void partitioning_approach<binary_search_type>::run(query_data& d, measurement& m);
template void partitioning_approach<radix_spline_type>::run(query_data& d, measurement& m);
template void partitioning_approach<no_op_type>::run(query_data& d, measurement& m);
