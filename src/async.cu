#include <sys/types.h>
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

#include "cuda_utils.cuh"
#include "cuda_allocator.hpp"
#include "numa_allocator.hpp"
#include "mmap_allocator.hpp"
#include "indexes.cuh"
#include "device_array.hpp"

#include "index_lookup_config.cuh"
#include "index_lookup_common.cuh"

#include "gpu_prefix_sum.hpp"
#include "partitioned_relation.hpp"
#include "utils.hpp"

#undef GPU_CACHE_LINE_SIZE
#include <numa-gpu/sql-ops/include/gpu_radix_partition.h>
#include <numa-gpu/sql-ops/cudautils/gpu_common.cu>
#include <numa-gpu/sql-ops/cudautils/radix_partition.cu>


static const int num_streams = 2;//2;
static const int block_size = 128;// 64;
static int grid_size = 0;//1;
static const uint32_t radix_bits = 12;// 10;
static const uint32_t ignore_bits = 4;//3;

template<class T> using device_allocator_t = cuda_allocator<T, cuda_allocation_type::device>;
template<class T> using device_index_allocator_t = cuda_allocator<T, cuda_allocation_type::zero_copy>;


struct PartitionedLookupArgs {
    // Input
    void* rel;
    uint32_t rel_length;
    uint32_t rel_padding_length;
    //uint64_t* rel_partition_offsets;
    unsigned long long* rel_partition_offsets;
    uint32_t* task_assignment;
    uint32_t radix_bits;
    uint32_t ignore_bits;
    // Output
    value_t* __restrict__ tids;
};

__global__ void join_kernel(float *x, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
    }
}

void dump_offsets(const partition_offsets& offsets) {
    auto h_offsets = offsets.offsets.to_host_accessible();
    std::cout << stringify(h_offsets.data(), h_offsets.data() + h_offsets.size()) << std::endl;
    auto h_local_offsets = offsets.local_offsets.to_host_accessible();
    std::cout << stringify(h_local_offsets.data(), h_local_offsets.data() + h_local_offsets.size()) << std::endl;
}

struct stream_state {
    cudaStream_t stream;

    size_t num_lookups;

    device_array_wrapper<int32_t> d_payloads;
    device_array_wrapper<index_key_t> d_dst_partition_attr;
    device_array_wrapper<int32_t> d_dst_payload_attrs;
    device_array_wrapper<value_t> d_dst_tids;
    device_array_wrapper<uint32_t> d_task_assignment;

    device_array_wrapper<ScanState<unsigned long long>> d_prefix_scan_state;

    partition_offsets partition_offsets_inst;
    partitioned_relation<Tuple<index_key_t, int32_t>> partitioned_relation_inst;

    std::unique_ptr<PrefixSumArgs> prefix_sum_and_copy_args;
    std::unique_ptr<RadixPartitionArgs> radix_partition_args;
    std::unique_ptr<PartitionedLookupArgs> partitioned_lookup_args;
};

std::unique_ptr<stream_state> create_stream_state(const index_key_t* d_lookup_keys, size_t num_lookups, value_t* d_dst_tids) {
    device_allocator_t<int> device_allocator;
    auto state = std::make_unique<stream_state>();
    CubDebugExit(cudaStreamCreate(&state->stream));

    state->num_lookups = num_lookups;

    // dummy payloads
    //state->d_payloads = create_device_array<int32_t>(num_lookups);

    // initialize payloads
    {
        std::vector<int32_t> payloads;
        payloads.resize(num_lookups);
        std::iota(payloads.begin(), payloads.end(), 0);
        state->d_payloads = create_device_array_from(payloads, device_allocator);
    }

    // allocate output arrays
    state->d_dst_partition_attr = create_device_array<index_key_t>(num_lookups);
    state->d_dst_payload_attrs = create_device_array<int32_t>(num_lookups);
    //state->d_dst_tids = create_device_array<value_t>(num_lookups);
    state->d_task_assignment = create_device_array<uint32_t>(grid_size + 1); // TODO check

    // see: device_exclusive_prefix_sum_initialize
    const auto prefix_scan_state_len = gpu_prefix_sum::state_size(grid_size, block_size);
    state->d_prefix_scan_state = create_device_array<ScanState<unsigned long long>>(prefix_scan_state_len);

    state->partition_offsets_inst = partition_offsets(grid_size, radix_bits, device_allocator);
    state->partitioned_relation_inst = partitioned_relation<Tuple<index_key_t, int32_t>>(num_lookups, grid_size, radix_bits, device_allocator);

    state->prefix_sum_and_copy_args = std::unique_ptr<PrefixSumArgs>(new PrefixSumArgs {
        // Inputs
        d_lookup_keys,
        num_lookups,
        0, // not used
        state->partitioned_relation_inst.padding_length(),
        radix_bits,
        ignore_bits,
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
        ignore_bits,
        state->partition_offsets_inst.local_offsets.data(),
        //state->partition_offsets_inst.offsets.data(),
        // State
        nullptr,
        nullptr,
        nullptr,
        0,
        // Outputs
        state->partitioned_relation_inst.relation.data()
    });

    state->partitioned_lookup_args = std::unique_ptr<PartitionedLookupArgs>(new PartitionedLookupArgs {
        state->partitioned_relation_inst.relation.data(),
        static_cast<uint32_t>(state->partitioned_relation_inst.relation.size()), // TODO check
        state->partitioned_relation_inst.padding_length(),
        state->partition_offsets_inst.offsets.data(),
        state->d_task_assignment.data(),
        radix_bits,
        ignore_bits,
        //state->d_dst_tids.data()
        d_dst_tids
    });

    return state;
}

template<class IndexStructureType>
__global__ void lookup_kernel(const IndexStructureType index_structure, unsigned n, const Tuple<index_key_t, int32_t>* __restrict__ relation, value_t* __restrict__ tids) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int i = index;
    uint32_t active_lanes = __ballot_sync(FULL_MASK, i < n);
    while (active_lanes) {
        bool active = i < n;
//printf("lookup %u\n", relation[i].key);
        auto tid = index_structure.cooperative_lookup(active, relation[i].key);
        if (active) {
            tids[i] = tid;
//            printf("tid %u\n", tid);
        }

        i += stride;
        active_lanes = __ballot_sync(FULL_MASK, i < n);
    }
}

__global__ void partitioned_lookup_assign_tasks(PartitionedLookupArgs args) {
    const auto fanout = 1U << args.radix_bits;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const uint32_t rel_size = args.rel_length - args.rel_padding_length*fanout;
        const uint32_t avg_task_size = (rel_size + gridDim.x - 1U) / gridDim.x;

        args.task_assignment[0] = 0U;
        uint32_t task_id = 1U;
        uint32_t task_size = 0U;
        for (uint32_t p = 0U; p < fanout && task_id < gridDim.x; ++p) {
            const uint32_t partition_upper = (p + 1U < fanout) ? args.rel_partition_offsets[p + 1U] - args.rel_padding_length : args.rel_length;
            const uint32_t partition_size = static_cast<uint32_t>(partition_upper - args.rel_partition_offsets[p]);

            task_size += partition_size;
            if (task_size >= avg_task_size) {
                args.task_assignment[task_id] = p + 1U;
// TODO
                task_size = 0U;
                task_id += 1;
            }
        }

        for (uint32_t i = task_id; i <= gridDim.x; ++i) {
            args.task_assignment[i] = fanout;
        }
    }
}

template<class TupleType, class IndexStructureType>
__global__ void partitioned_lookup_kernel(const IndexStructureType index_structure, const PartitionedLookupArgs args) {
    const auto fanout = 1U << args.radix_bits;


    for (uint32_t p = args.task_assignment[blockIdx.x]; p < args.task_assignment[blockIdx.x + 1U]; ++p) {
        const TupleType* __restrict__ relation = reinterpret_cast<const TupleType*>(args.rel) + args.rel_partition_offsets[p];

        const uint32_t partition_upper = (p + 1U < fanout) ? args.rel_partition_offsets[p + 1U] - args.rel_padding_length : args.rel_length;
        const uint32_t partition_size = static_cast<uint32_t>(partition_upper - args.rel_partition_offsets[p]);

#if 0
        // standard lookup implementation
        for (uint32_t i = threadIdx.x; i < partition_size; i += blockDim.x) {
            TupleType tuple = relation[i];
//printf("thread: %u i: %u lookup: %i\n", threadIdx.x, i, tuple.key);
            const auto tid = index_structure.lookup(tuple.key);
            args.tids[tuple.value] = tid;
        }
#else
        // cooperative lookup implementation
        for (uint32_t i = threadIdx.x; i < partition_size + 31; i += blockDim.x) {
//printf("thread: %u i: %u lookup: %i\n", threadIdx.x, i, tuple.key);
            const bool active = i < partition_size;
            TupleType tuple = active ? relation[i] : TupleType();
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


void dump_task_assignment(const stream_state& state) {
    const auto assignment = state.d_task_assignment.to_host_accessible();
    std::cout << "task assignment: " << stringify(assignment.data(), assignment.data() + assignment.size()) << std::endl;
}

template<class ResultVectorType>
bool validate_results(const std::vector<index_key_t>& lookup_keys, const ResultVectorType& tids) {
    const auto h_tids = tids.to_host_accessible();

    //std::cout << "tids: " << stringify(h_tids.data(), h_tids.data() + h_tids.size()) << std::endl;

    bool valid = true;
    for (size_t i = 0; i < lookup_keys.size(); ++i) {
        if (h_tids.data()[i] != lookup_keys[i]) {
            valid = false;
            std::cerr << "missmatch at: " << i << std::endl;
        }
    }
    std::cout << "validation done" << std::endl;
    return valid;
}


template<class IndexStructureType>
void run_on_stream(stream_state& state, IndexStructureType& index_structure, const cudaDeviceProp& device_properties) {
    const auto required_shared_mem_bytes = ((block_size + (block_size >> LOG2_NUM_BANKS)) + gpu_prefix_sum::fanout(radix_bits)) * sizeof(uint64_t);
    printf("required_shared_mem_bytes %lu\n", required_shared_mem_bytes);

    assert(required_shared_mem_bytes <= device_properties.sharedMemPerBlock);

    // prepare kernel arguments
    void* args[1];
    args[0] = state.prefix_sum_and_copy_args.get();

    CubDebugExit(cudaLaunchCooperativeKernel(
        (void*)gpu_contiguous_prefix_sum_int32,
        dim3(grid_size),
        dim3(block_size),
        args,
        required_shared_mem_bytes,
        state.stream
    ));
/*
cudaDeviceSynchronize();
auto r = state.partition_offsets_inst.offsets.to_host_accessible();
std::cout << "offsets: " << stringify(r.data(), r.data() + state.partition_offsets_inst.local_offsets.size()) << std::endl;
*/
    const auto required_shared_mem_bytes_2 = gpu_prefix_sum::fanout(radix_bits) * sizeof(uint32_t);

    //gpu_chunked_radix_partition_int32_int32<<<grid_size, block_size, required_shared_mem_bytes_2, state.stream>>>(*state.radix_partition_args);
    gpu_chunked_laswwc_radix_partition_int32_int32<<<grid_size, block_size, device_properties.sharedMemPerBlock, state.stream>>>(*state.radix_partition_args, device_properties.sharedMemPerBlock);
    //gpu_chunked_sswwc_radix_partition_v2_int32_int32<<<grid_size, block_size, device_properties.sharedMemPerBlock, state.stream>>>(*state.radix_partition_args, device_properties.sharedMemPerBlock);

/*
cudaDeviceSynchronize();
auto r2 = state.partitioned_relation_inst.relation.to_host_accessible();
std::cout << "result: " << stringify(r2.data(), r2.data() + state.partitioned_relation_inst.relation.size()) << std::endl;
*/
//dump_partitions(state);

    partitioned_lookup_assign_tasks<<<grid_size, 1, 0, state.stream>>>(*state.partitioned_lookup_args);

/*
cudaDeviceSynchronize();
dump_task_assignment(state);
*/
partitioned_lookup_kernel<Tuple<index_key_t, int32_t>><<<grid_size, block_size, 0, state.stream>>>(index_structure.device_index, *state.partitioned_lookup_args);
//cudaDeviceSynchronize();
}

int main(int argc, char** argv) {
    double zipf_factor = 1.25;
    auto num_elements = default_num_elements;
    size_t num_lookups = default_num_lookups;
    if (argc > 1) {
        std::string::size_type sz;
        num_elements = std::stod(argv[1], &sz);
    }
    std::cout << "index size: " << num_elements << std::endl;


    if (grid_size == 0) {
        int num_sms;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        grid_size = num_sms;
    }


    // generate datasets
    std::vector<index_key_t, host_allocator_t<index_key_t>> indexed, lookup_keys;
    indexed.resize(num_elements);
    lookup_keys.resize(num_lookups);
    generate_datasets<index_key_t>(dataset_type::dense, max_bits, indexed, lookup_pattern_type::uniform, zipf_factor, lookup_keys);
    //std::cout << stringify(lookup_keys.begin(), lookup_keys.end());

    // create gpu accessible vectors
    indexed_allocator_t indexed_allocator;
    auto d_indexed = create_device_array_from(indexed, indexed_allocator);
    lookup_keys_allocator_t lookup_keys_allocator;
    auto d_lookup_keys = create_device_array_from(lookup_keys, lookup_keys_allocator);
    auto index = build_index<index_key_t, index_type>(indexed, d_indexed.data());
    auto d_dst_tids = create_device_array<value_t>(num_lookups);


    // fetch device properties
    cudaDeviceProp device_properties;
    CubDebugExit(cudaGetDeviceProperties(&device_properties, 0));
    std::cout << "sharedMemPerBlock: " << device_properties.sharedMemPerBlock << std::endl;

    size_t remaining = num_lookups;
    size_t max_stream_portion = num_lookups / num_streams;
    const index_key_t* d_stream_lookup_keys = d_lookup_keys.data();
    value_t* d_stream_tids = d_dst_tids.data();

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




    auto start_ts = std::chrono::high_resolution_clock::now();
    for (const auto& state : stream_states) {
        run_on_stream(*state, *index, device_properties);
    }
    cudaDeviceSynchronize();
    const auto stop_ts = std::chrono::high_resolution_clock::now();
    const auto rt = std::chrono::duration_cast<std::chrono::microseconds>(stop_ts - start_ts).count()/1000.;
    std::cout << "Kernel time: " << rt << " ms\n";
    std::cout << "GPU MOps: " << (num_lookups/1e6)/(rt/1e3) << std::endl;



    validate_results(lookup_keys, d_dst_tids);

    cudaDeviceReset();

    return 0;
}
