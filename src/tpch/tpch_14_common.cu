#include "tpch_14_common.cuh"

#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <memory>

#include <numa-gpu/sql-ops/include/gpu_common.h>
#include "gpu_radix_partition.cuh"

#include "config.hpp"
#include "common.hpp"
#include "device_properties.hpp"
#include "indexes.cuh"
#include "utils.hpp"
#include "LinearProbingHashTable.cuh"
#include "measuring.hpp"
#include "partitioned_relation.hpp"
#include "tpch_14_ij.cuh"
#include "tpch_14_ij_partitioning.cuh"

index_type_enum parse_index_type(const std::string& index_name) {
    if (index_name == "btree") {
        return index_type_enum::btree;
    } else if (index_name == "harmonia") {
        return index_type_enum::harmonia;
    } else if (index_name == "lower_bound") {
        return index_type_enum::lower_bound;
    } else if (index_name == "radix_spline") {
        return index_type_enum::radix_spline;
    } else if (index_name == "no_op") {
        return index_type_enum::no_op;
    } else {
        throw std::runtime_error("unknown index type");
    }
}

struct query_data {
    Database db;

    std::unique_ptr<abstract_index<indexed_t>> index_structure;

    unsigned lineitem_size;
    lineitem_table_plain_t* lineitem_device;
    std::unique_ptr<lineitem_table_plain_t> lineitem_device_ptrs;

    unsigned part_size;
    part_table_plain_t* part_device;
    std::unique_ptr<part_table_plain_t> part_device_ptrs;

    void load_database() {
        const auto& config = get_experiment_config();

        load_tables(db, config.db_path);
        if (config.sort_indexed_relation) {
            printf("sorting part relation...\n");
            sort_relation(db.part);
        }
        lineitem_size = db.lineitem.l_orderkey.size();
        part_size = db.part.p_partkey.size();

        {
            using namespace std;
            const auto start = chrono::high_resolution_clock::now();
            device_table_allocator<int> allocator;
            std::tie(lineitem_device, lineitem_device_ptrs) = migrate_relation(db.lineitem, allocator);
            std::tie(part_device, part_device_ptrs) = migrate_relation(db.part, allocator);
            const auto finish = chrono::high_resolution_clock::now();
            const auto d = chrono::duration_cast<chrono::milliseconds>(finish - start).count();
            std::cout << "migration time: " << d << " ms\n";
        }
    }

    void create_index() {
        const auto& config = get_experiment_config();

        // allocate index structure
        switch (parse_index_type(config.index_type)) {
            case index_type_enum::btree:
                index_structure = std::make_unique<btree_type>();
                break;
            case index_type_enum::harmonia:
                index_structure = std::make_unique<harmonia_type>();
                break;
            case index_type_enum::lower_bound:
                index_structure = std::make_unique<lower_bound_type>();
                break;
            case index_type_enum::radix_spline:
                index_structure = std::make_unique<radix_spline_type>();
                break;
            case index_type_enum::no_op:
                index_structure = std::make_unique<no_op_type>();
                break;
            default:
                assert(false);
        }

        const auto view = make_vector_view(db.part.p_partkey);
        index_structure->construct(view, part_device_ptrs->p_partkey);
        printf("index size: %lu bytes\n", index_structure->memory_consumption());
    }
};

template<class T>
void print_results(const T& d_mutable_state) {
    // Fetch result
    const auto r = d_mutable_state.to_host_accessible();
    const auto& state = r.data()[0];
    auto numerator = state.global_numerator;
    auto denominator = state.global_denominator;
    printf("numerator: %ld denominator: %ld\n", (long)numerator, (long)denominator);

    numerator *= 1'000;
    denominator /= 1'000;
    int64_t result = 100*numerator/denominator;
    printf("query result: %ld.%ld\n", result/1'000'000, result%1'000'000);
}

struct abstract_approach_dispatcher {
    virtual void run(query_data& d, index_type_enum index_type) const = 0;
};

template<template<class T> class Func>
struct approach_dispatcher : public abstract_approach_dispatcher {
    void run(query_data& d, index_type_enum index_type) const override {
        switch (index_type) {
            case index_type_enum::btree:
                Func<btree_type>()(d);
                break;
            case index_type_enum::harmonia:
                Func<harmonia_type>()(d);
                break;
            case index_type_enum::lower_bound:
                Func<lower_bound_type>()(d);
                break;
            case index_type_enum::radix_spline:
                Func<radix_spline_type>()(d);
                break;
            case index_type_enum::no_op:
                Func<no_op_type>()(d);
                break;
            default:
                assert(false);
        }
    }
};

template<class IndexType>
struct my_approach {
    void operator()(query_data& d) {
        printf("my approach %s\n", type_name<IndexType>::value());
    }
};

using device_ht_t = LinearProbingHashTable<uint32_t, size_t>::DeviceHandle;

__global__ void hj_build_kernel(size_t n, const part_table_plain_t* part, device_ht_t ht);

__global__ void hj_probe_kernel(size_t n, const part_table_plain_t* __restrict__ part, const lineitem_table_plain_t* __restrict__ lineitem, device_ht_t ht);

template<class IndexType>
struct hj_approach {
    void operator()(query_data& d) {
        const auto& config = get_experiment_config();

        LinearProbingHashTable<uint32_t, size_t> ht(d.part_size);
        int num_blocks = (d.part_size + config.block_size - 1) / config.block_size;
        hj_build_kernel<<<num_blocks, config.block_size>>>(d.part_size, d.part_device, ht.deviceHandle);

        //num_blocks = 32*num_sms;
        num_blocks = (d.lineitem_size + config.block_size - 1) / config.block_size;
        hj_probe_kernel<<<num_blocks, config.block_size>>>(d.lineitem_size, d.part_device, d.lineitem_device, ht.deviceHandle);
        cudaDeviceSynchronize();

// TODO        print_results();
    }
};

template<class IndexStructureType>
__global__ void ij_plain_kernel(const lineitem_table_plain_t* __restrict__ lineitem, const unsigned lineitem_size, const part_table_plain_t* __restrict__ part, IndexStructureType index_structure);

template<class IndexType>
struct ij_plain_approach {
    void operator()(query_data& d) {
        const auto& config = get_experiment_config();

        struct ij_mutable_state mutable_state;

        auto d_mutable_state = create_device_array<ij_mutable_state>(1);
        target_memcpy<device_exclusive_allocator<int>>()(d_mutable_state.data(), &mutable_state, sizeof(ij_mutable_state));

        const ij_args args {
            // Inputs
            d.lineitem_device,
            d.lineitem_size,
            d.part_device,
            d.part_size,
            // State and outputs
            d_mutable_state.data()
        };

        const int num_blocks = (d.lineitem_size + config.block_size - 1) / config.block_size;

        IndexType& index_structure = *static_cast<IndexType*>(d.index_structure.get());
        ij_plain_kernel<<<num_blocks, config.block_size>>>(args, index_structure.device_index);
        cudaDeviceSynchronize();

        print_results(d_mutable_state);
    }
};

// Pipelined Blockwise Sorting
template<class IndexType>
struct ij_pbws_approach {
    void operator()(query_data& d) {
//        const auto& config = get_experiment_config();

        enum { BLOCK_THREADS = 256, ITEMS_PER_THREAD = 10 }; // TODO optimize

        const int num_blocks = 4 * get_device_properties(0).multiProcessorCount;
        const int buffer_size = num_blocks*BLOCK_THREADS*(ITEMS_PER_THREAD + 1);

        auto d_l_extendedprice_buffer = create_device_array<numeric_raw_t>(buffer_size);
        auto d_l_discount_buffer = create_device_array<numeric_raw_t>(buffer_size);

        struct ij_mutable_state mutable_state;
        mutable_state.l_extendedprice_buffer = d_l_extendedprice_buffer.data();
        mutable_state.l_discount_buffer = d_l_discount_buffer.data();

        auto d_mutable_state = create_device_array<ij_mutable_state>(1);
        target_memcpy<device_exclusive_allocator<int>>()(d_mutable_state.data(), &mutable_state, sizeof(ij_mutable_state));

        const ij_args args {
            // Inputs
            d.lineitem_device,
            d.lineitem_size,
            d.part_device,
            d.part_size,
            // State and outputs
            d_mutable_state.data()
        };

        IndexType& index_structure = *static_cast<IndexType*>(d.index_structure.get());
        ij_pbws<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(args, index_structure.device_index);
        cudaDeviceSynchronize();

        print_results(d_mutable_state);
    }
};

/*
void run_two_phase_ij_plain() {
    join_entry* join_entries;
    cudaMalloc(&join_entries, sizeof(join_entry)*lineitem_size);

    const auto kernelStart = std::chrono::high_resolution_clock::now();

    int num_blocks = (part_size + block_size - 1) / block_size;
    ij_lookup_kernel<<<num_blocks, block_size>>>(lineitem_device, lineitem_size, index_structure.device_index, join_entries);
    cudaDeviceSynchronize();

    decltype(output_index) matches;
    cudaError_t error = cudaMemcpyFromSymbol(&matches, output_index, sizeof(matches), 0, cudaMemcpyDeviceToHost);
    assert(error == cudaSuccess);
    //printf("join matches: %u\n", matches);

    num_blocks = (lineitem_size + block_size - 1) / block_size;
    ij_join_kernel<<<num_blocks, block_size>>>(lineitem_device, part_device, join_entries, matches);
    cudaDeviceSynchronize();

    const auto kernelStop = std::chrono::high_resolution_clock::now();
    const auto kernelTime = std::chrono::duration_cast<std::chrono::microseconds>(kernelStop - kernelStart).count()/1000.;
    std::cout << "kernel time: " << kernelTime << " ms\n";
}
*/

// Non-Pipelined Lookup First Plain approach
template<class IndexType>
struct ij_nplfplain_approach {
    static constexpr double selectivity_est = 0.02; // actual floating point result: 0.0126612694262745

    void operator()(query_data& d) {
        const auto& config = get_experiment_config();

        auto d_join_entries = create_device_array<join_entry>(selectivity_est*d.lineitem_size);

        struct ij_mutable_state mutable_state;
        mutable_state.join_entries = d_join_entries.data();

        auto d_mutable_state = create_device_array<ij_mutable_state>(1);
        target_memcpy<device_exclusive_allocator<int>>()(d_mutable_state.data(), &mutable_state, sizeof(ij_mutable_state));

        const ij_args args {
            // Inputs
            d.lineitem_device,
            d.lineitem_size,
            d.part_device,
            d.part_size,
            // State and outputs
            d_mutable_state.data()
        };

        IndexType& index_structure = *static_cast<IndexType*>(d.index_structure.get());

        int num_blocks = (d.part_size + config.block_size - 1) / config.block_size;
        ij_lookup_kernel<<<num_blocks, config.block_size>>>(args, index_structure.device_index);

        num_blocks = (d.lineitem_size + config.block_size - 1) / config.block_size;
        ij_join_kernel<<<num_blocks, config.block_size>>>(args);
        cudaDeviceSynchronize();

        print_results(d_mutable_state);
    }
};

template<class IndexType>
struct ij_partitioning_approach {
    using payload_type = std::remove_pointer_t<decltype(lineitem_table_plain_t::l_extendedprice)>;

    static constexpr unsigned num_streams = 1;
    static constexpr unsigned oversubscription_factor = 2;
    static constexpr unsigned radix_bits = 10; // TODO
    static constexpr unsigned ignore_bits = 4; // TODO
    static constexpr double selectivity_est = 0.02; // actual floating point result: 0.0126612694262745

    size_t buffer_size;
    int grid_size;

    std::unique_ptr<partitioned_ij_scan_args> partitioned_ij_scan_args_inst;
    device_array_wrapper<partitioned_ij_scan_mutable_state> d_mutable_state;
    // materialized attributes
    device_array_wrapper<indexed_t> d_l_partkey_materialized;
    device_array_wrapper<payload_type> d_summand_materialized;

    struct stream_state {
        cudaStream_t stream;

        device_array_wrapper<uint32_t> d_task_assignment;
        device_array_wrapper<ScanState<unsigned long long>> d_prefix_scan_state;

        partition_offsets partition_offsets_inst;
        partitioned_relation<partitioned_tuple_type> partitioned_relation_inst;

        // Kernel arguments
        std::unique_ptr<PrefixSumArgs> prefix_sum_args;
        std::unique_ptr<RadixPartitionArgs> radix_partition_args;
        std::unique_ptr<partitioned_consumer_assign_tasks_args> partitioned_consumer_assign_tasks_args_inst;
        std::unique_ptr<partitioned_ij_lookup_args> partitioned_ij_lookup_args_inst;
        device_array_wrapper<partitioned_ij_lookup_mutable_state> d_mutable_state;
    } stream_states[num_streams];

    size_t buffer_size_upper_bound(const query_data& d) const {
        const size_t lineitem_size = d.db.lineitem.l_partkey.size();
        return static_cast<size_t>(std::ceil(selectivity_est * lineitem_size));
    }

    size_t fetch_materialized_size() {
        const auto r = d_mutable_state.to_host_accessible();
        return r.data()[0].materialized_size;
    }

    void init(query_data& d) {
        cuda_allocator<uint8_t, cuda_allocation_type::device> device_allocator;

        buffer_size = buffer_size_upper_bound(d);

        d_l_partkey_materialized = create_device_array<indexed_t>(buffer_size);
        d_summand_materialized = create_device_array<payload_type>(buffer_size);

        const partitioned_ij_scan_mutable_state mutable_state {
            // State
            d_l_partkey_materialized.data(),
            d_summand_materialized.data(),
            0u
        };
        d_mutable_state = create_device_array<partitioned_ij_scan_mutable_state>(1);
        target_memcpy<decltype(device_allocator)>()(d_mutable_state.data(), &mutable_state, sizeof(partitioned_ij_scan_mutable_state));

        partitioned_ij_scan_args_inst = std::unique_ptr<partitioned_ij_scan_args>(new partitioned_ij_scan_args {
            // Inputs
            d.lineitem_device,
            d.lineitem_size,
            // State and outputs
            d_mutable_state.data()
        });
    }

    void phase1(query_data& d) {
        const auto& config = get_experiment_config();
        const auto& device_properties = get_device_properties(0);

        const int num_blocks = (d.lineitem_size + config.block_size - 1) / config.block_size;
        //partitioned_ij_scan<<<num_blocks, config.block_size>>>(*partitioned_ij_scan_args_inst);
        partitioned_ij_scan_refill<<<num_blocks, config.block_size, device_properties.sharedMemPerBlock>>>(*partitioned_ij_scan_args_inst);
        cudaDeviceSynchronize();
        printf("phase1 done\n");
    }

    void phase2(query_data& d) {
        const auto materialized_size = fetch_materialized_size();

        size_t remaining = materialized_size;
        size_t max_stream_portion = (materialized_size + num_streams) / num_streams;
        indexed_t* d_indexed = d_l_partkey_materialized.data();
        payload_type* d_payloads = d_summand_materialized.data();

        // create streams
        for (unsigned i = 0; i < num_streams; ++i) {
            size_t stream_portion = std::min(remaining, max_stream_portion);
            remaining -= stream_portion;
            printf("stream portion: %lu\n", stream_portion);
            init_stream_state(i, d, d_indexed, stream_portion, d_payloads);

            d_indexed += stream_portion;
            d_payloads += stream_portion;
        }

        const auto& device_properties = get_device_properties(0);
        IndexType& index_structure = *static_cast<IndexType*>(d.index_structure.get());
        for (const auto& state : stream_states) {
            //void run_on_stream(stream_state& state, IndexStructureType& index_structure, const cudaDeviceProp& device_properties)

            run_on_stream(state, index_structure, device_properties);
        }
        cudaDeviceSynchronize();
    }

    void init_stream_state(const unsigned state_num, const query_data& d, indexed_t* d_indexed, const uint32_t stream_portion, payload_type* d_payloads) {
        const auto& config = get_experiment_config();
        cuda_allocator<uint8_t, cuda_allocation_type::device> device_allocator;

        grid_size = get_device_properties(0).multiProcessorCount * oversubscription_factor / num_streams;

        auto& state = stream_states[state_num];
        CubDebugExit(cudaStreamCreate(&state.stream));

        // allocate output arrays
        state.d_task_assignment = create_device_array<uint32_t>(grid_size + 1); // TODO check

        // see: device_exclusive_prefix_sum_initialize
        const auto prefix_scan_state_len = gpu_prefix_sum::state_size(grid_size, config.block_size);
        state.d_prefix_scan_state = create_device_array<ScanState<unsigned long long>>(prefix_scan_state_len);

        state.partition_offsets_inst = partition_offsets(grid_size, radix_bits, device_allocator);
        state.partitioned_relation_inst = partitioned_relation<partitioned_tuple_type>(stream_portion, grid_size, radix_bits, device_allocator);

        state.prefix_sum_args = std::unique_ptr<PrefixSumArgs>(new PrefixSumArgs {
            // Inputs
            d_indexed,
            stream_portion,
            0, // not used
            state.partitioned_relation_inst.padding_length(),
            radix_bits,
            ignore_bits,
            // State
            state.d_prefix_scan_state.data(),
            state.partition_offsets_inst.local_offsets.data(),
            // Outputs
            state.partition_offsets_inst.offsets.data()
        });

        state.radix_partition_args = std::unique_ptr<RadixPartitionArgs>(new RadixPartitionArgs {
            // Inputs
            d_indexed,
            d_payloads,
            stream_portion,
            state.partitioned_relation_inst.padding_length(),
            radix_bits,
            ignore_bits,
            state.partition_offsets_inst.local_offsets.data(),
            //state.partition_offsets_inst.offsets.data(),
            // State
            nullptr, // tmp_partition_offsets - used by gpu_chunked_sswwc_radix_partition_v2
            nullptr, // l2_cache_buffers - only used by gpu_chunked_sswwc_radix_partition_v2g
            nullptr, // device_memory_buffers - only used by gpu_chunked_hsswwc_* kernels
            0, // device_memory_buffer_bytes - only used by gpu_chunked_hsswwc_* kernels
            // Outputs
            state.partitioned_relation_inst.relation.data()
        });

        state.partitioned_consumer_assign_tasks_args_inst = std::unique_ptr<partitioned_consumer_assign_tasks_args>(new partitioned_consumer_assign_tasks_args {
            // Inputs
            static_cast<uint32_t>(state.partitioned_relation_inst.relation.size()), // TODO check
            state.partitioned_relation_inst.padding_length(),
            state.partition_offsets_inst.offsets.data(),
            radix_bits,
            // Outputs
            state.d_task_assignment.data()
        });

        // Initialize lookup-kernel arguments
        const partitioned_ij_lookup_mutable_state mutable_state {
            // Outputs
            0l,
            0l
        };
        state.d_mutable_state = create_device_array<partitioned_ij_lookup_mutable_state>(1);
        target_memcpy<decltype(device_allocator)>()(d_mutable_state.data(), &mutable_state, sizeof(partitioned_ij_lookup_mutable_state));

        state.partitioned_ij_lookup_args_inst = std::unique_ptr<partitioned_ij_lookup_args>(new partitioned_ij_lookup_args {
            // Inputs
            d.part_device,
            state.partitioned_relation_inst.relation.data(),
            static_cast<uint32_t>(state.partitioned_relation_inst.relation.size()), // TODO check
            state.partitioned_relation_inst.padding_length(),
            state.partition_offsets_inst.offsets.data(),
            state.d_task_assignment.data(),
            radix_bits,
            // State and outputs
            state.d_mutable_state.data()
        });
    }

    void run_on_stream(const stream_state& state, IndexType& index_structure, const cudaDeviceProp& device_properties) {
        const auto& config = get_experiment_config();

        // calculate prefix sum kernel shared memory requirement
        const auto required_shared_mem_bytes = ((config.block_size + (config.block_size >> LOG2_NUM_BANKS)) + gpu_prefix_sum::fanout(radix_bits)) * sizeof(uint64_t);
#ifdef DEBUG_INTERMEDIATE_STATE
        printf("required_shared_mem_bytes %lu\n", required_shared_mem_bytes);
#endif
        assert(required_shared_mem_bytes <= device_properties.sharedMemPerBlock);

        // prepare kernel arguments
        void* args[1];
        args[0] = state.prefix_sum_args.get();

        // calculate prefix sum
        CubDebugExit(cudaLaunchCooperativeKernel(
            (void*)gpu_contiguous_prefix_sum_int64,
            dim3(grid_size),
            dim3(config.block_size),
            args,
            required_shared_mem_bytes,
            state.stream
        ));
#ifdef DEBUG_INTERMEDIATE_STATE
        cudaDeviceSynchronize();
        auto r = state.partition_offsets_inst.offsets.to_host_accessible();
        std::cout << "offsets: " << stringify(r.data(), r.data() + state.partition_offsets_inst.local_offsets.size()) << std::endl;
#endif

        // calculate radix partition kernel shared memory requirement
        const auto required_shared_mem_bytes_2 = gpu_prefix_sum::fanout(radix_bits) * sizeof(uint32_t);

        gpu_chunked_radix_partition_int64_int64<<<grid_size, config.block_size, device_properties.sharedMemPerBlock, state.stream>>>(*state.radix_partition_args);
        //gpu_chunked_laswwc_radix_partition_int64_int64<<<grid_size, config.block_size, device_properties.sharedMemPerBlock, state.stream>>>(*state.radix_partition_args, device_properties.sharedMemPerBlock);
        //gpu_chunked_sswwc_radix_partition_v2_int64_int64<<<grid_size, config.block_size, device_properties.sharedMemPerBlock, state.stream>>>(*state.radix_partition_args, device_properties.sharedMemPerBlock);

#ifdef DEBUG_INTERMEDIATE_STATE
        cudaDeviceSynchronize();
        auto r2 = state.partitioned_relation_inst.relation.to_host_accessible();
        std::cout << "result: " << stringify(r2.data(), r2.data() + state.partitioned_relation_inst.relation.size()) << std::endl;
        dump_partitions(state);
#endif

        partitioned_consumer_assign_tasks<<<grid_size, 1, 0, state.stream>>>(*state.partitioned_consumer_assign_tasks_args_inst);
#ifdef DEBUG_INTERMEDIATE_STATE
        cudaDeviceSynchronize();
        dump_task_assignment(state);
#endif

        partitioned_ij_lookup<<<grid_size, config.block_size, 0, state.stream>>>(*state.partitioned_ij_lookup_args_inst, index_structure.device_index);
    }

    void operator()(query_data& d) {
        init(d);
        phase1(d);
        phase2(d);
        cudaDeviceSynchronize();
        // collect results
        int64_t numerator = 0;
        int64_t denominator = 0;
        for (const auto& stream_state : stream_states) {
            const auto r = stream_state.d_mutable_state.to_host_accessible();
            const auto& state = r.data()[0];
            numerator += state.global_numerator;
            denominator += state.global_denominator;
        }
        printf("numerator: %ld denominator: %ld\n", (long)numerator, (long)denominator);

        numerator *= 1'000;
        denominator /= 1'000;
        int64_t result = 100*numerator/denominator;
        printf("query result: %ld.%ld\n", result/1'000'000, result%1'000'000);
    }
};

// see https://stackoverflow.com/questions/8016780/undefined-reference-to-static-constexpr-char
template<class IndexType>
constexpr unsigned ij_partitioning_approach<IndexType>::num_streams;

template<class IndexType>
constexpr unsigned ij_partitioning_approach<IndexType>::oversubscription_factor;

//static const std::map<std::string, std::unique_ptr<abstract_approach_dispatcher>> approaches {
static const std::map<std::string, std::shared_ptr<abstract_approach_dispatcher>> approaches {
    { "hj", std::make_shared<approach_dispatcher<hj_approach>>() },
    { "ij_plain", std::make_shared<approach_dispatcher<ij_plain_approach>>() },
    { "ij_pbws", std::make_shared<approach_dispatcher<ij_pbws_approach>>() },
    { "ij_nplfplain", std::make_shared<approach_dispatcher<ij_nplfplain_approach>>() },
    { "ij_partitioning", std::make_shared<approach_dispatcher<ij_partitioning_approach>>() }
};

static measuring::experiment_description create_experiment_description() {
    const auto& config = get_experiment_config();

    measuring::experiment_description r;
    r.name = "tpch_query14";
    r.approach = config.approach;
    std::vector<std::pair<std::string, std::string>> other = {
        std::make_pair(std::string("device"), std::string(get_device_properties(0).name)),
        std::make_pair(std::string("db_path"), config.db_path),
        std::make_pair(std::string("sort_indexed_relation"), tmpl_to_string(config.sort_indexed_relation)),
        std::make_pair(std::string("block_size"), tmpl_to_string(config.block_size)),

        // allocators:
        std::make_pair(std::string("host_allocator"), std::string(type_name<host_allocator<int>>::value())),
        std::make_pair(std::string("device_index_allocator"), std::string(type_name<device_index_allocator<int>>::value())),
        std::make_pair(std::string("device_table_allocator"), std::string(type_name<device_table_allocator<int>>::value()))
    };

    if (r.approach != "hj") {
        other.emplace_back(std::string("index_type"), config.index_type);
        other.emplace_back(std::string("prefetch_index"), tmpl_to_string(config.prefetch_index));
    }

    if (r.approach == "ij_partitioning") {
        other.emplace_back(std::string("num_stream"), tmpl_to_string(ij_partitioning_approach<no_op_type>::num_streams));
        other.emplace_back(std::string("oversubscription_factor"), tmpl_to_string(ij_partitioning_approach<no_op_type>::oversubscription_factor));
    }

    r.other.swap(other);

    return r;
}

void execute_approach(std::string approach_name) {
    auto& config = get_experiment_config();

    query_data qd;
    qd.load_database();
    if (config.approach != "hj") {
        qd.create_index();
    }

    const auto experiment_desc = create_experiment_description();
    index_type_enum index_type = parse_index_type(config.index_type);
    measure(experiment_desc, [&]() {
        approaches.at(approach_name)->run(qd, index_type);
    });
}
