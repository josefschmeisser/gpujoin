#include "index_lookup.cuh"

#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <memory>

#include <oneapi/tbb/parallel_sort.h>

#include "index_lookup_config.hpp"
#include "index_lookup_common.cuh"
#include "index_lookup_partitioning.cuh"
#include "device_properties.hpp"
#include "indexes.hpp"
#include "measuring.hpp"

#ifdef NRDC
#include "index_lookup_partitioning.cu"
#endif

using namespace measuring;


query_data::query_data() {
    auto& config = get_experiment_config();

    // generate datasets
    printf("generating datasets...\n");
    indexed.resize(config.num_elements);
    lookup_keys.resize(config.num_lookups);
    generate_datasets<index_key_t>(config.dataset, config.max_bits, indexed, config.lookup_pattern, config.zipf_factor, lookup_keys);
    if (config.sorted_lookups) {
        printf("sorting lookups...\n");
        oneapi::tbb::parallel_sort(lookup_keys.begin(), lookup_keys.end());
    }
    //std::cout << "lookups: " << stringify(lookup_keys.begin(), lookup_keys.end()) << std::endl;

    // indexed is guaranteed to be sorted
    dataset_max_bits = static_cast<unsigned>(std::log2(indexed.back()));
    if (config.partitioning_approach_dynamic_bit_range) {
        config.partitioning_approach_ignore_bits = (dataset_max_bits > radix_bits) ? dataset_max_bits - radix_bits : config.partitioning_approach_ignore_bits;
        printf("config.partitioning_approach_ignore_bits: %d\n", config.partitioning_approach_ignore_bits);
    }

    // allocate result vector
    d_tids = create_device_array<value_t>(config.num_lookups);

    // create gpu accessible vectors
    indexed_allocator_t indexed_allocator;
    d_indexed = create_device_array_from(indexed, indexed_allocator);
    lookup_keys_allocator_t lookup_keys_allocator;
    d_lookup_keys = create_device_array_from(lookup_keys, lookup_keys_allocator);

    // finalize state
    printf("generating index...\n");
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
        case index_type_enum::binary_search:
            index_structure = build_index<index_key_t, binary_search_type>(indexed, d_indexed.data());
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
        if (h_tids_raw[i] > indexed.size()) {
            std::cerr << "invalid tid: " << h_tids_raw[i] << ", at " << i << " from " << lookup_keys.size() << std::endl;
            return false;
        }
        if (lookup_keys[i] != indexed[h_tids_raw[i]]) {
            std::cerr << "lookup_keys[" << i << "]: " << lookup_keys[i] << "indexed[h_tids[" << i << "]]: " << indexed[h_tids_raw[i]] << std::endl;
            return false;
        }
    }
    printf("validation complete\n");

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
            case index_type_enum::binary_search:
                Func<binary_search_type>()(d);
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
    }
};

template<class IndexType>
struct blockwise_sorting_approach {
    void operator()(query_data& d) {
        const auto& config = get_experiment_config();

        const int num_blocks = 3 * get_device_properties(0).multiProcessorCount;; // TODO optimize
        printf("numblocks: %d\n", num_blocks);

        printf("executing kernel...\n");
        IndexType& index_structure = *static_cast<IndexType*>(d.index_structure.get());
        if (config.block_size != 256) {
            std::cerr << "invalid block size for this approach" << std::endl;
            throw 0;
        }
        lookup_kernel_with_sorting_v1<256, 4><<<num_blocks, 256>>>(index_structure.device_index, d.lookup_keys.size(), d.d_lookup_keys.data(), d.d_tids.data(), d.dataset_max_bits);
        cudaDeviceSynchronize();
    }
};

//static const std::map<std::string, std::unique_ptr<abstract_approach_dispatcher>> approaches {
static const std::map<std::string, std::shared_ptr<abstract_approach_dispatcher>> approaches {
    { "plain", std::make_shared<approach_dispatcher<plain_approach>>() },
    { "bws", std::make_shared<approach_dispatcher<blockwise_sorting_approach>>() },
    { "partitioning", std::make_shared<approach_dispatcher<partitioning_approach>>() }
};

static void add_index_configuration_description(std::vector<std::pair<std::string, std::string>>& pairs, const query_data& qd) {
    const auto& config = get_experiment_config();

    switch (parse_index_type(config.index_type)) {
        case index_type_enum::btree:
            pairs.emplace_back("index_lookup_algorithm", std::string(btree_type::index_configuration_t::cooperative_lookup_algorithm_type::name()));
            break;
        case index_type_enum::binary_search:
            pairs.emplace_back("index_search_algorithm", std::string(binary_search_type::index_configuration_t::cooperative_search_algorithm_type::name()));
            break;
        case index_type_enum::radix_spline:
            pairs.emplace_back("index_search_algorithm", std::string(radix_spline_type::index_configuration_t::cooperative_search_algorithm_type::name()));
            break;
    }

    pairs.emplace_back("index_size", "!int64 " + std::to_string(qd.index_structure->memory_consumption()));
}

static void create_common_experiment_description_pairs_2(std::vector<std::pair<std::string, std::string>>& pairs) {
    const auto& config = get_experiment_config();

    pairs.emplace_back(std::string("device"), std::string(get_device_properties(0).name));
    pairs.emplace_back(std::string("index_type"), config.index_type);
    pairs.emplace_back(std::string("dataset"), tmpl_to_string(config.dataset));
    pairs.emplace_back(std::string("lookup_pattern"), tmpl_to_string(config.lookup_pattern));
    pairs.emplace_back(std::string("num_elements"), "!int64 " + std::to_string(config.num_elements));
    pairs.emplace_back(std::string("num_lookups"), "!int64 " + std::to_string(config.num_lookups));
    pairs.emplace_back(std::string("sorted_lookups"), std::to_string(config.sorted_lookups));
        // allocators:
    pairs.emplace_back(std::string("host_allocator"), std::string(type_name<host_allocator_t<int>>::value()));
    pairs.emplace_back(std::string("device_index_allocator"), std::string(type_name<device_index_allocator<int>>::value()));
    pairs.emplace_back(std::string("indexed_allocator"), std::string(type_name<indexed_allocator_t>::value()));
    pairs.emplace_back(std::string("lookup_keys_allocator"), std::string(type_name<lookup_keys_allocator_t>::value()));

    if (config.dataset == dataset_type::sparse) {
        pairs.emplace_back(std::string("max_bits"), std::to_string(config.max_bits));
    }

    if (config.lookup_pattern == lookup_pattern_type::zipf) {
        pairs.emplace_back(std::string("zipf_factor"), std::to_string(config.zipf_factor));
    }

    if (config.approach == "partitioning") {
        pairs.emplace_back(std::string("partitioning_approach_ignore_bits"), std::to_string(config.partitioning_approach_ignore_bits));
        pairs.emplace_back(std::string("partitioning_approach_dynamic_bit_range"), std::to_string(config.partitioning_approach_dynamic_bit_range));
    }
}

static measuring::experiment_description create_experiment_description(const query_data& qd) {
    const auto& config = get_experiment_config();

    experiment_description r;
    r.name = "plain_lookup";
    r.approach = config.approach;

    create_common_experiment_description_pairs_2(r.other);
    add_index_configuration_description(r.other, qd);

    return r;
}

void execute_approach(std::string approach_name) {
    auto& config = get_experiment_config();

    query_data qd;

    const auto experiment_desc = create_experiment_description(qd);
    index_type_enum index_type = parse_index_type(config.index_type);
    auto validator = [&qd]() {
        return qd.validate_results();
    };
    measure(experiment_desc, [&](auto& measurement) {
        approaches.at(approach_name)->run(qd, index_type);
    }, validator);
}

void execute_benchmark_scenario(std::string scenario) {
    const auto& config = get_experiment_config();
    execute_approach(config.approach);
}

int main(int argc, char** argv) {
    parse_options(argc, argv);
    const auto& config = get_experiment_config();

    // set-up the measuring utility
    auto& measuring_config = measuring::get_settings();
    if (!config.output_file.empty()) {
        measuring_config.stdout_only = false;
        measuring_config.dest_file = config.output_file;
    } else {
        measuring_config.stdout_only = true;
    }
    measuring_config.repetitions = 10;
/*
    if (config.execute_predefined_scenario) {
        execute_benchmark_scenario();
        return;
    }*/

    execute_benchmark_scenario(config.scenario);

#if 0
    const auto experiment_desc = create_experiment_description();
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
