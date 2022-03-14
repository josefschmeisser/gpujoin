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
static const int block_size = 64;// 128;// 64;
static const int grid_size = 2;//1;
static const uint32_t radix_bits = 6;// 10;
static const uint32_t ignore_bits = 0;//3;

template<class T> using device_allocator_t = cuda_allocator<T, cuda_allocation_type::device>;
template<class T> using device_index_allocator_t = cuda_allocator<T, cuda_allocation_type::zero_copy>;



//template<class IndexStructureType>
struct PartitionedLookupArgs {
    // Input
//    IndexStructureType index_structure;
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
//printf("num_lookups: %lu\n", num_lookups);
    CubDebugExit(cudaStreamCreate(&state->stream));
/*
auto wrapper = device_array_wrapper<index_key_t>::create_reference_only(const_cast<index_key_t*>(d_lookup_keys), num_lookups);
auto r = wrapper.to_host_accessible();
*/

#if 0
std::vector<index_key_t> tmp;
tmp.resize(num_lookups);
cudaMemcpy(tmp.data(), d_lookup_keys, num_lookups*sizeof(index_key_t), cudaMemcpyDeviceToHost);
std::cout << "input:" << stringify(tmp.begin(), tmp.end()) << std::endl;
#endif

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

/*
template<class IndexStructureType, class TupleType>
__global__ void partitioned_lookup_assign_tasks(PartitionedLookupArgs<IndexStructureType>& args) {
*/
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
//__global__ void partitioned_lookup_kernel(PartitionedLookupArgs<IndexStructureType>& args) {
__global__ void partitioned_lookup_kernel(const IndexStructureType index_structure, const PartitionedLookupArgs args) {
#if 0
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int i = index;
    uint32_t active_lanes = __ballot_sync(FULL_MASK, i < n);
    while (active_lanes) {
        bool active = i < n;
printf("lookup %u\n", relation[i].key);
        auto tid = index_structure.cooperative_lookup(active, relation[i].key);
        if (active) {
            tids[i] = tid;
            printf("tid %lu\n", tid);
        }

        i += stride;
        active_lanes = __ballot_sync(FULL_MASK, i < n);
    }
#endif
    const auto fanout = 1U << args.radix_bits;


    for (uint32_t p = args.task_assignment[blockIdx.x]; p < args.task_assignment[blockIdx.x + 1U]; ++p) {
        const TupleType* __restrict__ relation = reinterpret_cast<const TupleType*>(args.rel) + args.rel_partition_offsets[p];

        const uint32_t partition_upper = (p + 1U < fanout) ? args.rel_partition_offsets[p + 1U] - args.rel_padding_length : args.rel_length;
        const uint32_t partition_size = static_cast<uint32_t>(partition_upper - args.rel_partition_offsets[p]);

        for (uint32_t i = threadIdx.x; i < partition_size; i += blockDim.x) {
            TupleType tuple = relation[i];
//printf("thread: %u i: %u lookup: %i\n", threadIdx.x, i, tuple.key);
            const auto tid = index_structure.lookup(tuple.key);
            args.tids[tuple.value] = tid;
        }
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

    std::cout << "tids: " << stringify(h_tids.data(), h_tids.data() + h_tids.size()) << std::endl;

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
cudaDeviceSynchronize();
auto r = state.partition_offsets_inst.offsets.to_host_accessible();
std::cout << "offsets: " << stringify(r.data(), r.data() + state.partition_offsets_inst.offsets.size()) << std::endl;

    const auto required_shared_mem_bytes_2 = gpu_prefix_sum::fanout(radix_bits) * sizeof(uint32_t);

    //gpu_chunked_radix_partition_int32_int32<<<grid_size, block_size, required_shared_mem_bytes_2, state.stream>>>(*state.radix_partition_args);
    gpu_chunked_laswwc_radix_partition_int32_int32<<<grid_size, block_size, device_properties.sharedMemPerBlock, state.stream>>>(*state.radix_partition_args, device_properties.sharedMemPerBlock);
    //gpu_chunked_sswwc_radix_partition_v2_int32_int32<<<grid_size, block_size, device_properties.sharedMemPerBlock, state.stream>>>(*state.radix_partition_args, device_properties.sharedMemPerBlock);


cudaDeviceSynchronize();
auto r2 = state.partitioned_relation_inst.relation.to_host_accessible();
std::cout << "result: " << stringify(r2.data(), r2.data() + state.partitioned_relation_inst.relation.size()) << std::endl;

//dump_partitions(state);

    partitioned_lookup_assign_tasks<<<grid_size, 1, 0, state.stream>>>(*state.partitioned_lookup_args);


cudaDeviceSynchronize();
dump_task_assignment(state);

//    lookup_kernel<<<grid_size, block_size, 4*1024, state.stream>>>(index_structure.device_index, state.num_lookups, state.partitioned_relation_inst.relation.data(), state.d_dst_tids.data());
partitioned_lookup_kernel<Tuple<index_key_t, int32_t>><<<grid_size, block_size, 0, state.stream>>>(index_structure.device_index, *state.partitioned_lookup_args);
cudaDeviceSynchronize();

//validate_results(state);

}

int main(int argc, char** argv) {
    double zipf_factor = 1.25;
    auto num_elements = default_num_elements;
    size_t num_lookups = 256;// default_num_lookups;
    if (argc > 1) {
        std::string::size_type sz;
        num_elements = std::stod(argv[1], &sz);
    }
    std::cout << "index size: " << num_elements << std::endl;

    // generate datasets
    std::vector<index_key_t, host_allocator_t<index_key_t>> indexed, lookup_keys;
    indexed.resize(num_elements);
    lookup_keys.resize(num_lookups);
    generate_datasets<index_key_t, index_type>(dataset_type::dense, max_bits, indexed, lookup_pattern_type::uniform, zipf_factor, lookup_keys);
std::cout << stringify(lookup_keys.begin(), lookup_keys.end());

    // create gpu accessible vectors
    indexed_allocator_t indexed_allocator;
    auto d_indexed = create_device_array_from(indexed, indexed_allocator);
    lookup_keys_allocator_t lookup_keys_allocator;
    auto d_lookup_keys = create_device_array_from(lookup_keys, lookup_keys_allocator);
    auto index = build_index<index_key_t, index_type>(indexed, d_indexed.data());
    auto d_dst_tids = create_device_array<value_t>(num_lookups);

/*
auto r = d_lookup_keys.to_host_accessible();
std::cout << "result:" << stringify(r.data(), r.data() + num_lookups) << std::endl;
return 0;
*/
    // fetch device properties
    cudaDeviceProp device_properties;
    CubDebugExit(cudaGetDeviceProperties(&device_properties, 0));
    std::cout << "sharedMemPerBlock: " << device_properties.sharedMemPerBlock << std::endl;

#if 0
    device_allocator_t<int> device_allocator;

    // dummy payloads
    auto d_payloads = create_device_array<int32_t>(num_lookups);

    // allocate output arrays
    auto dst_partition_attr = create_device_array<index_key_t>(num_lookups);
    auto dst_payload_attrs = create_device_array<int32_t>(num_lookups);

    //ScanState<unsigned long long>* prefix_scan_state; // see: device_exclusive_prefix_sum_initialize
    const auto prefix_scan_state_len = gpu_prefix_sum::state_size(grid_size, block_size);
    auto prefix_scan_state = create_device_array<ScanState<unsigned long long>>(prefix_scan_state_len);

    partition_offsets offsets(grid_size, radix_bits, device_allocator);
    //partitioned_relation<index_key_t> partitioned_relation_inst(num_lookups, grid_size, radix_bits, device_allocator);
    partitioned_relation<Tuple<int32_t, int32_t>> partitioned_relation_inst(num_lookups, grid_size, radix_bits, device_allocator);

    PrefixSumArgs prefix_sum_and_copy_args {
        // Inputs
        d_lookup_keys.data(),
        num_lookups,
        0, // not used
        partitioned_relation_inst.padding_length(),
        radix_bits,
        ignore_bits,
        // State
        prefix_scan_state.data(),
        offsets.local_offsets.data(),
        // Outputs
        offsets.offsets.data()
    };

    //__host__ ​cudaError_t cudaLaunchCooperativeKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream ) 
    /*
    template <typename K, typename V>
    __device__ void gpu_contiguous_prefix_sum_and_copy_with_payload(args)
    */
    cudaStream_t scan_stream;
    CubDebugExit(cudaStreamCreate(&scan_stream));

    const auto required_shared_mem_bytes = ((block_size + (block_size >> LOG2_NUM_BANKS)) + gpu_prefix_sum::fanout(radix_bits)) * sizeof(uint64_t);
    printf("required_shared_mem_bytes %lu\n", required_shared_mem_bytes);
    //const void* func = &gpu_contiguous_prefix_sum_and_copy_with_payload<int, int>;
    assert(required_shared_mem_bytes <= device_properties.sharedMemPerBlock);

    // prepare kernel arguments
    //auto d_prefix_sum_and_copy_args = create_device_array_from(reinterpret_cast<const uint8_t*>(&prefix_sum_and_copy_args), sizeof(prefix_sum_and_copy_args));
    void* args[1];
    args[0] = &prefix_sum_and_copy_args;// d_prefix_sum_and_copy_args.data();

    CubDebugExit(cudaLaunchCooperativeKernel(
        //func,
//        (void*)gpu_contiguous_prefix_sum_and_copy_with_payload_int32_int32,
        //(void*)gpu_chunked_prefix_sum_int32,
        (void*)gpu_contiguous_prefix_sum_int32,
        dim3(grid_size),
        dim3(block_size),
        args,
        required_shared_mem_bytes,
        scan_stream
    ));
    cudaDeviceSynchronize();
    printf("gpu_contiguous_prefix_sum_and_copy_with_payload_int32_int32 done\n");

dump_offsets(offsets);
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
        //dst_partition_attr.data(),
        //dst_payload_attrs.data(),
        d_lookup_keys.data(),
        d_payloads.data(),
        num_lookups,
        partitioned_relation_inst.padding_length(),
        radix_bits,
        ignore_bits,
//        offsets.local_offsets.data(),
        offsets.offsets.data(),
        // State
        nullptr,
        nullptr,
        nullptr,
        0,
        // Outputs
        partitioned_relation_inst.relation.data()
    };

    const auto required_shared_mem_bytes_2 = gpu_prefix_sum::fanout(radix_bits) * sizeof(uint32_t);

    /*
    template <typename K, typename V>
    __device__ void gpu_chunked_laswwc_radix_partition(RadixPartitionArgs &args, uint32_t shared_mem_bytes);
    */

    //gpu_chunked_laswwc_radix_partition<<<1, 64>>>(args, );

//gpu_chunked_laswwc_radix_partition_int32_int32<<<grid_size, block_size, device_properties.sharedMemPerBlock, scan_stream>>>(radix_partition_args, device_properties.sharedMemPerBlock);

    gpu_chunked_radix_partition_int32_int32<<<grid_size, block_size, required_shared_mem_bytes_2, scan_stream>>>(radix_partition_args);
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
#endif




    size_t remaining = num_lookups;
    size_t max_stream_portion = num_lookups / num_streams;
    const index_key_t* d_stream_lookup_keys = d_lookup_keys.data();
    value_t* d_stream_tids = d_dst_tids.data();
/*
auto r = d_lookup_keys.to_host_accessible();
std::cout << "input:" << stringify(r.data(), r.data() + num_lookups) << std::endl;
*/
    //printf("estimated partition size: %lu\n", partition_size);

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

    for (const auto& state : stream_states) {
        run_on_stream(*state, *index, device_properties);
    }
    cudaDeviceSynchronize();

    validate_results(lookup_keys, d_dst_tids);

    cudaDeviceReset();

    return 0;
}
