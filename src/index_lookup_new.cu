#include "index_lookup.cuh"

#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <memory>

#include "index_lookup_config.hpp"
#include "index_lookup_common.cuh"
#include "index_lookup_partitioning.cuh"
#include "device_properties.hpp"
#include "measuring.hpp"

using namespace measuring;


query_data::query_data() {
    const auto& config = get_experiment_config();

    // generate datasets
    printf("generating datasets...\n");
    indexed.resize(config.num_elements);
    lookup_keys.resize(config.num_lookups);
    generate_datasets<index_key_t>(dataset_type::sparse, config.max_bits, indexed, lookup_pattern_type::uniform, config.zipf_factor, lookup_keys);

    // allocate result vector
    d_tids = create_device_array<value_t>(config.num_lookups);

    // create gpu accessible vectors
    indexed_allocator_t indexed_allocator;
    auto d_indexed = create_device_array_from(indexed, indexed_allocator);
    lookup_keys_allocator_t lookup_keys_allocator;
    d_lookup_keys = create_device_array_from(lookup_keys, lookup_keys_allocator);

    // finalize state
    create_index();
}

void query_data::create_index() {
    const auto& config = get_experiment_config();

    // allocate index structure
    switch (parse_index_type(config.index_type)) {
        case index_type_enum::btree:
            index_structure = build_index<index_key_t, btree_type>(indexed, d_indexed.data());
            break;
        case index_type_enum::harmonia:
            index_structure = build_index<index_key_t, harmonia_type>(indexed, d_indexed.data());
            break;
        case index_type_enum::lower_bound:
            index_structure = build_index<index_key_t, lower_bound_type>(indexed, d_indexed.data());
            break;
        case index_type_enum::radix_spline:
            index_structure = build_index<index_key_t, radix_spline_type>(indexed, d_indexed.data());
            break;
        case index_type_enum::no_op:
            index_structure = build_index<index_key_t, no_op_type>(indexed, d_indexed.data());
            break;
        default:
            assert(false);
    }
}

bool query_data::validate_results() {
    auto h_tids = d_tids.to_host_accessible();
    auto h_tids_raw = h_tids.data();

    // validate results
    printf("validating results...\n");
    for (unsigned i = 0; i < lookup_keys.size(); ++i) {
        if (lookup_keys[i] != indexed[h_tids_raw[i]]) {
            printf("lookup_keys[%u]: %u indexed[h_tids[%u]]: %u\n", i, lookup_keys[i], i, indexed[h_tids_raw[i]]);
            fflush(stdout);
            return false;
        }
    }

    return true;
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

/*
template<class IndexType>
struct hj_approach {
    void operator()(query_data& d) {
        const auto& config = get_experiment_config();

        LinearProbingHashTable<uint32_t, size_t> ht(d.part_size);

        hj_mutable_state mutable_state {
            ht.deviceHandle
        };

        auto d_mutable_state = create_device_array<hj_mutable_state>(1);
        target_memcpy<device_exclusive_allocator<int>>()(d_mutable_state.data(), &mutable_state, sizeof(mutable_state));

        const hj_args args {
            // Inputs
            d.lineitem_device,
            d.lineitem_size,
            d.part_device,
            d.part_size,
            // State and outputs
            d_mutable_state.data()
        };

        int num_blocks = (d.part_size + config.block_size - 1) / config.block_size;
        hj_build_kernel<<<num_blocks, config.block_size>>>(args);

        num_blocks = (d.lineitem_size + config.block_size - 1) / config.block_size;
        hj_probe_kernel<<<num_blocks, config.block_size>>>(args);
        cudaDeviceSynchronize();

        print_results(d_mutable_state);
    }
};*/

#if false
template<class IndexStructureType>
auto run_lookup_benchmark(const measuring::experiment_description& experiment_desc, IndexStructureType& index_structure, const index_key_t* d_lookup_keys, unsigned num_lookup_keys) {
    const auto& config = get_experiment_config();

    int num_blocks;
    if /*constexpr*/ (!config.partitial_sorting) {
        num_blocks = (num_lookup_keys + blockSize - 1) / blockSize;
    } else {
        int num_sms;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        num_blocks = num_sms*3; // TODO
    }
    printf("numblocks: %d\n", num_blocks);

    // create result array
    value_t* d_tids;
    cudaMalloc(&d_tids, num_lookup_keys*sizeof(value_t));

    printf("executing kernel...\n");

    measure(experiment_desc, [&]() {
        if /*constexpr*/ (!config.partitial_sorting) {
            lookup_kernel<<<num_blocks, blockSize>>>(index_structure.device_index, num_lookup_keys, d_lookup_keys, d_tids);
        } else {
            lookup_kernel_with_sorting_v1<blockSize, 4><<<num_blocks, blockSize>>>(index_structure.device_index, num_lookup_keys, d_lookup_keys, d_tids, config.max_bits);
        }
        cudaDeviceSynchronize();
    });

    // transfer results
    std::unique_ptr<value_t[]> h_tids(new value_t[num_lookup_keys]);
    cudaMemcpy(h_tids.get(), d_tids, num_lookup_keys*sizeof(value_t), cudaMemcpyDeviceToHost);

    cudaFree(d_tids);

    return std::move(h_tids);
}
#endif


template<class IndexType>
struct plain_approach {
    void operator()(query_data& d) {
        const auto& config = get_experiment_config();

        const int num_blocks = (config.num_lookups + config.block_size - 1) / config.block_size;
        printf("numblocks: %d\n", num_blocks);

        printf("executing kernel...\n");
        IndexType& index_structure = *static_cast<IndexType*>(d.index_structure.get());
        lookup_kernel<<<num_blocks, config.block_size>>>(index_structure.device_index, d.lookup_keys.size(), d.d_lookup_keys.data(), d.d_tids.data());

        cudaDeviceSynchronize();

/*
        // transfer results
        std::unique_ptr<value_t[]> h_tids(new value_t[num_lookup_keys]);
        cudaMemcpy(h_tids.get(), d_tids, num_lookup_keys*sizeof(value_t), cudaMemcpyDeviceToHost);

        cudaFree(d_tids);

        return std::move(h_tids);*/
    }
};

template<class IndexType>
struct blockwise_sorting_approach {
    void operator()(query_data& d) {
        const auto& config = get_experiment_config();

        const int num_blocks = 3 * get_device_properties(0).multiProcessorCount;; // TODO
        printf("numblocks: %d\n", num_blocks);

        printf("executing kernel...\n");
        IndexType& index_structure = *static_cast<IndexType*>(d.index_structure.get());
        if (config.block_size != 256) {
            throw 0;
        }
        lookup_kernel_with_sorting_v1<256, 4><<<num_blocks, 256>>>(index_structure.device_index, d.lookup_keys.size(), d.d_lookup_keys.data(), d.d_tids.data(), config.max_bits);
        cudaDeviceSynchronize();

/*
        // transfer results
        std::unique_ptr<value_t[]> h_tids(new value_t[num_lookup_keys]);
        cudaMemcpy(h_tids.get(), d_tids, num_lookup_keys*sizeof(value_t), cudaMemcpyDeviceToHost);

        cudaFree(d_tids);

        return std::move(h_tids);*/
    }
};

/*
template<class IndexType>
struct partitioning_approach {
    void operator()(query_data& d) {
    }
};*/

//static const std::map<std::string, std::unique_ptr<abstract_approach_dispatcher>> approaches {
static const std::map<std::string, std::shared_ptr<abstract_approach_dispatcher>> approaches {
    { "plain", std::make_shared<approach_dispatcher<plain_approach>>() },
    { "bws", std::make_shared<approach_dispatcher<blockwise_sorting_approach>>() },
    { "partitioning", std::make_shared<approach_dispatcher<partitioning_approach>>() }
};

std::vector<std::pair<std::string, std::string>> create_common_experiment_description_pairs_2() {
    const auto& config = get_experiment_config();

    std::vector<std::pair<std::string, std::string>> r = {
        std::make_pair(std::string("device"), std::string(get_device_properties(0).name)),
        std::make_pair(std::string("index_type"), config.index_type),
        std::make_pair(std::string("dataset"), tmpl_to_string(config.dataset)),
        std::make_pair(std::string("lookup_pattern"), tmpl_to_string(config.lookup_pattern)),
        std::make_pair(std::string("num_elements"), std::to_string(config.num_elements)),
        std::make_pair(std::string("num_lookups"), std::to_string(config.num_lookups)),
        // allocators:
        std::make_pair(std::string("host_allocator"), std::string(type_name<host_allocator_t<int>>::value())),
        std::make_pair(std::string("device_index_allocator"), std::string(type_name<device_index_allocator<int>>::value())),
        std::make_pair(std::string("indexed_allocator"), std::string(type_name<indexed_allocator_t>::value())),
        std::make_pair(std::string("lookup_keys_allocator"), std::string(type_name<lookup_keys_allocator_t>::value()))
    };

    if (config.dataset == dataset_type::sparse) {
        r.push_back(std::make_pair(std::string("max_bits"), std::to_string(config.max_bits)));
    }

    if (config.lookup_pattern == lookup_pattern_type::zipf) {
        r.push_back(std::make_pair(std::string("zipf_factor"), std::to_string(config.zipf_factor)));
    }

    return r;
}

static measuring::experiment_description create_experiment_description() {
    const auto& config = get_experiment_config();
/*
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
*/

    experiment_description r;
    r.name = "plain_lookup";
    r.approach = config.approach;
    r.other = create_common_experiment_description_pairs_2();
    return r;
}

void execute_approach(std::string approach_name) {
    auto& config = get_experiment_config();

    query_data qd;

    const auto experiment_desc = create_experiment_description();
    index_type_enum index_type = parse_index_type(config.index_type);
    measure(experiment_desc, [&]() {
        approaches.at(approach_name)->run(qd, index_type);
    });
}

/*
int main(int argc, char** argv) {
    parse_options(argc, argv);
    const auto& config = get_experiment_config();

    // set-up the measuring utility
    auto& measuring_config = measuring::get_settings();
    measuring_config.dest_file = "tpch_14_results.yml";
    measuring_config.repetitions = 1;
    measuring_config.stdout_only = true;
    // TODO
    //const auto experiment_desc = create_experiment_description();

    execute_approach(config.approach);
    return 0;
}
*/

void execute_benchmark_scenario(std::string scenario) {
    const auto& config = get_experiment_config();
    // TODO
    execute_approach(config.approach);
}

int main(int argc, char** argv) {
/*
    double zipf_factor = 1.25;
    auto num_elements = default_num_elements;
    auto num_lookups = default_num_lookups;
    if (argc > 1) {
        std::string::size_type sz;
        num_elements = std::stod(argv[1], &sz);
    }
    std::cout << "index size: " << num_elements << std::endl;
*/
    parse_options(argc, argv);
    const auto& config = get_experiment_config();

    // set-up the measuring utility
    auto& measuring_config = measuring::get_settings();
    measuring_config.dest_file = "index_scan_results.yml";
    measuring_config.stdout_only = true;
    measuring_config.repetitions = 1;
    const auto experiment_desc = create_experiment_description();
/*
    if (config.execute_predefined_scenario) {
        execute_benchmark_scenario();
        return;
    }*/

    execute_benchmark_scenario(config.scenario);

#if 0
    // TODO port
    std::unique_ptr<value_t[]> h_tids;
    if /*constexpr*/ (activeLanes < 32) {
        assert(false); // TODO
    } else {
        auto result = run_lookup_benchmark(experiment_desc, *index, d_lookup_keys.data(), lookup_keys.size());
        h_tids.swap(result);
    }
#endif

    return 0;
}
