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

static const int num_streams = 2;
static const int block_size = 128;// 64;
static int grid_size = 0;
//static const uint32_t radix_bits = 11;// 10;
//static const uint32_t ignore_bits = 4;//3;

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
    uint32_t* task_assignments;
    uint32_t radix_bits;
    uint32_t ignore_bits;
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
    device_array_wrapper<uint32_t> d_task_assignments;

    device_array_wrapper<ScanState<unsigned long long>> d_prefix_scan_state;

    partition_offsets partition_offsets_inst;
    partitioned_relation<rel_tuple_t> partitioned_relation_inst;

    std::unique_ptr<PrefixSumArgs> prefix_sum_and_copy_args;
    std::unique_ptr<RadixPartitionArgs> radix_partition_args;
    std::unique_ptr<PartitionedLookupArgs> partitioned_lookup_args;
};

std::unique_ptr<stream_state> create_stream_state(const index_key_t* d_lookup_keys, uint32_t num_lookups, value_t* d_dst_tids) {
    const auto& config = get_experiment_config();
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
    state->d_task_assignments = create_device_array<uint32_t>(grid_size + 1); // TODO check

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
        state->d_task_assignments.data(),
        radix_bits,
        config.partitioning_approach_ignore_bits,
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

// TODO replace
__global__ void partitioned_lookup_assign_tasks(PartitionedLookupArgs args) {
    const auto fanout = 1U << args.radix_bits;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const uint32_t rel_size = args.rel_length - args.rel_padding_length*fanout;
        const uint32_t avg_task_size = (rel_size + gridDim.x - 1U) / gridDim.x;

        args.task_assignments[0] = 0U;
        uint32_t task_id = 1U;
        uint32_t task_size = 0U;
        for (uint32_t p = 0U; p < fanout && task_id < gridDim.x; ++p) {
            const uint32_t partition_upper = (p + 1U < fanout) ? args.rel_partition_offsets[p + 1U] - args.rel_padding_length : args.rel_length;
            const uint32_t partition_size = static_cast<uint32_t>(partition_upper - args.rel_partition_offsets[p]);

            task_size += partition_size;
            if (task_size >= avg_task_size) {
                args.task_assignments[task_id] = p + 1U;
// TODO
                task_size = 0U;
                task_id += 1;
            }
        }

        for (uint32_t i = task_id; i <= gridDim.x; ++i) {
            args.task_assignments[i] = fanout;
        }
    }
}

template<class TupleType, class IndexStructureType>
__global__ void partitioned_lookup_kernel(const IndexStructureType index_structure, const PartitionedLookupArgs args) {
    const auto fanout = 1U << args.radix_bits;

    for (uint32_t p = args.task_assignments[blockIdx.x]; p < args.task_assignments[blockIdx.x + 1U]; ++p) {
        const TupleType* __restrict__ relation = reinterpret_cast<const TupleType*>(args.rel) + args.rel_partition_offsets[p];

        const uint32_t partition_upper = (p + 1U < fanout) ? args.rel_partition_offsets[p + 1U] - args.rel_padding_length : args.rel_length;
        const uint32_t partition_size = static_cast<uint32_t>(partition_upper - args.rel_partition_offsets[p]);
        const uint32_t loop_limit = (partition_size + warpSize - 1) & ~(warpSize - 1); // round to next multiple of warpSize

#if 0
        // standard lookup implementation
        for (uint32_t i = threadIdx.x; i < partition_size; i += blockDim.x) {
            const TupleType tuple = relation[i];
            const auto tid = index_structure.lookup(tuple.key);
            args.tids[tuple.value] = tid;
        }
#else
        // cooperative lookup implementation
        for (uint32_t i = threadIdx.x; i < loop_limit; i += blockDim.x) {
            const bool active = i < partition_size;
            const TupleType tuple = active ? relation[i] : TupleType();
            const auto tid = index_structure.cooperative_lookup(active, tuple.key);
            if (active) {
                args.tids[tuple.value] = tid;
            }
        }
#endif
    }
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
    const auto assignment = state.d_task_assignments.to_host_accessible();
    std::cout << "task assignment: " << stringify(assignment.data(), assignment.data() + assignment.size()) << std::endl;
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

    partitioned_lookup_kernel<rel_tuple_t><<<grid_size, block_size, 0, state.stream>>>(index_structure.device_index, *state.partitioned_lookup_args);
}

template<class IndexType>
void partitioning_approach<IndexType>::operator()(query_data& d, measurement& m) {
    printf("partitioning_approach\n");
    const auto& config = get_experiment_config();
    const auto& device_properties = get_device_properties(0);

    if (grid_size == 0) {
        grid_size = device_properties.multiProcessorCount;
    }

    if (config.num_lookups >= std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("config.num_lookups >= std::numeric_limits<uint32_t>::max()");
    }
    size_t remaining = config.num_lookups;
    size_t max_stream_portion = (config.num_lookups + num_streams) / num_streams;
    //printf("ALIGN_BYTES: %u\n", ALIGN_BYTES);
    max_stream_portion = (max_stream_portion + ALIGN_BYTES - 1) & -ALIGN_BYTES;
    const index_key_t* d_stream_lookup_keys = d.d_lookup_keys.data();
    value_t* d_stream_tids = d.d_tids.data();

    std::vector<std::unique_ptr<stream_state>> stream_states;

    // create streams
    for (unsigned i = 0; i < num_streams; ++i) {
        size_t stream_portion = std::min(remaining, max_stream_portion);
        remaining -= stream_portion;
        printf("stream portion: %lu\n", stream_portion);
        auto state = create_stream_state(d_stream_lookup_keys, stream_portion, d_stream_tids);
        stream_states.push_back(std::move(state));

        d_stream_lookup_keys += stream_portion;
        d_stream_tids += stream_portion;
    }

    IndexType& index_structure = *static_cast<IndexType*>(d.index_structure.get());
    for (const auto& state : stream_states) {
        run_on_stream(*state, index_structure, device_properties);
    }
    cudaDeviceSynchronize();
}

template void partitioning_approach<btree_type>::operator()(query_data& d, measurement& m);
template void partitioning_approach<harmonia_type>::operator()(query_data& d, measurement& m);
template void partitioning_approach<binary_search_type>::operator()(query_data& d, measurement& m);
template void partitioning_approach<radix_spline_type>::operator()(query_data& d, measurement& m);
template void partitioning_approach<no_op_type>::operator()(query_data& d, measurement& m);
