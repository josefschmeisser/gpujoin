#include "tpch_14_common.cuh"

#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <memory>

#include "config.hpp"
#include "common.hpp"
#include "device_properties.hpp"
#include "indexes.cuh"
#include "utils.hpp"
#include "LinearProbingHashTable.cuh"
#include "measuring.hpp"

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

static measuring::experiment_description create_experiment_description() {
    const auto& config = get_experiment_config();

    measuring::experiment_description r;
    r.name = "tpch_query14";
    r.approach = config.approach;
    std::vector<std::pair<std::string, std::string>> other = {
        std::make_pair(std::string("device"), std::string(get_device_properties(0).name)),
        std::make_pair(std::string("index_type"), config.index_type),
        std::make_pair(std::string("db_path"), config.db_path),
        std::make_pair(std::string("prefetch_index"), tmpl_to_string(config.prefetch_index)),
        std::make_pair(std::string("sort_indexed_relation"), tmpl_to_string(config.sort_indexed_relation)),
        std::make_pair(std::string("block_size"), tmpl_to_string(config.block_size)),

        // allocators:
        std::make_pair(std::string("host_allocator"), std::string(type_name<host_allocator<int>>::value())),
        std::make_pair(std::string("device_index_allocator"), std::string(type_name<device_index_allocator<int>>::value())),
        std::make_pair(std::string("device_table_allocator"), std::string(type_name<device_table_allocator<int>>::value()))
    };
    r.other.swap(other);

    return r;
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
    }
};

template<class IndexStructureType>
__global__ void test_kernel();

template<class IndexType>
struct streamed_ij_approach {
    static constexpr unsigned num_streams = 2;

    struct stream_state {
        cudaStream_t stream;
/*
        device_array_wrapper<int32_t> d_payloads;
        device_array_wrapper<index_key_t> d_dst_partition_attr;
        device_array_wrapper<int32_t> d_dst_payload_attrs;
        device_array_wrapper<value_t> d_dst_tids;
        device_array_wrapper<uint32_t> d_task_assignment;
*/
        device_array_wrapper<ScanState<unsigned long long>> d_prefix_scan_state;

        partition_offsets partition_offsets_inst;
        partitioned_relation<Tuple<index_key_t, int32_t>> partitioned_relation_inst;

        std::unique_ptr<PrefixSumArgs> prefix_sum_and_copy_args;
        std::unique_ptr<RadixPartitionArgs> radix_partition_args;
        std::unique_ptr<PartitionedLookupArgs> partitioned_lookup_args;
    } stream_states[num_streams];

    streamed_ij_approach() {
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
            state->d_task_assignment.data(),
            radix_bits,
            ignore_bits,
            //state->d_dst_tids.data()
            d_dst_tids
        });

        return state;
    }

    void operator()(query_data& d) {
        using namespace std;

        enum { BLOCK_THREADS = 256, ITEMS_PER_THREAD = 10 }; // TODO optimize

#if 0
        int num_sms;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        int num_blocks = num_sms*4; // TODO

        auto left = left_pipeline(db);

        uint32_t* d_l_partkey_buffer = left.l_partkey_buffer_guard.data();
        int64_t* d_l_extendedprice_buffer = left.l_extendedprice_buffer_guard.data();
        int64_t* d_l_discount_buffer = left.l_discount_buffer_guard.data();

        const auto kernelStart = std::chrono::high_resolution_clock::now();

        ij_join_finalization_kernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(d_l_partkey_buffer, d_l_extendedprice_buffer, d_l_discount_buffer, left.size, part_device, part_size, index_structure.device_index);
        cudaDeviceSynchronize();

        const auto kernelStop = std::chrono::high_resolution_clock::now();
        const auto kernelTime = std::chrono::duration_cast<std::chrono::microseconds>(kernelStop - kernelStart).count()/1000.;
        std::cout << "kernel time: " << kernelTime << " ms\n";
#endif

/*
        //ij_join_streamed_btree<<<1, 32>>>(d.index_structure->device_index);
        test_kernel<IndexType><<<1, 32>>>();
        cudaDeviceSynchronize();*/

        //////////////////////////////////////////////////////////////////////////////////////////////////


        // calculate prefix sum kernel shared memory requirement
        const auto required_shared_mem_bytes = ((block_size + (block_size >> LOG2_NUM_BANKS)) + gpu_prefix_sum::fanout(radix_bits)) * sizeof(uint64_t);
#ifdef DEBUG_INTERMEDIATE_STATE
        printf("required_shared_mem_bytes %lu\n", required_shared_mem_bytes);
#endif
        assert(required_shared_mem_bytes <= device_properties.sharedMemPerBlock);

        // prepare kernel arguments
        void* args[1];
        args[0] = state.prefix_sum_and_copy_args.get();

        // calculate prefix sum
        CubDebugExit(cudaLaunchCooperativeKernel(
            (void*)gpu_contiguous_prefix_sum_int32,
            dim3(grid_size),
            dim3(block_size),
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

        //gpu_chunked_radix_partition_int32_int32<<<grid_size, block_size, required_shared_mem_bytes_2, state.stream>>>(*state.radix_partition_args);
        //gpu_chunked_laswwc_radix_partition_int32_int32<<<grid_size, block_size, device_properties.sharedMemPerBlock, state.stream>>>(*state.radix_partition_args, device_properties.sharedMemPerBlock);
        gpu_chunked_sswwc_radix_partition_v2_int32_int32<<<grid_size, block_size, device_properties.sharedMemPerBlock, state.stream>>>(*state.radix_partition_args, device_properties.sharedMemPerBlock);

#ifdef DEBUG_INTERMEDIATE_STATE
        cudaDeviceSynchronize();
        auto r2 = state.partitioned_relation_inst.relation.to_host_accessible();
        std::cout << "result: " << stringify(r2.data(), r2.data() + state.partitioned_relation_inst.relation.size()) << std::endl;
        dump_partitions(state);
#endif

        partitioned_lookup_assign_tasks<<<grid_size, 1, 0, state.stream>>>(*state.partitioned_lookup_args);
#ifdef DEBUG_INTERMEDIATE_STATE
        cudaDeviceSynchronize();
        dump_task_assignment(state);
#endif

        partitioned_lookup_kernel<Tuple<index_key_t, int32_t>><<<grid_size, block_size, 0, state.stream>>>(index_structure.device_index, *state.partitioned_lookup_args);
    }
};

//static const std::map<std::string, std::unique_ptr<abstract_approach_dispatcher>> approaches {
static const std::map<std::string, std::shared_ptr<abstract_approach_dispatcher>> approaches {
    { "option1", std::make_shared<approach_dispatcher<my_approach>>() },
    { "hj", std::make_shared<approach_dispatcher<hj_approach>>() },
    { "streamed_ij", std::make_shared<approach_dispatcher<streamed_ij_approach>>() }
};

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
