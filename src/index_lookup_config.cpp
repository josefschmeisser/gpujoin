#include "index_lookup_config.hpp"
#include "index_lookup_config.tpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>

#include "thirdparty/cxxopts.hpp"

template<>
std::string tmpl_to_string(const dataset_type& v) {
    return (v == dataset_type::dense) ? "dense" : "sparse";
}

template<>
std::string tmpl_to_string(const lookup_pattern_type& v) {
    switch (v) {
        case lookup_pattern_type::uniform:
            return "uniform";
        case lookup_pattern_type::uniform_unique:
            return "uniform_unique";
        case lookup_pattern_type::zipf:
            return "zipf";
        default:
            assert(false);
    }
}

experiment_config& get_experiment_config() {
    return singleton<experiment_config>::instance();
}

void parse_options(int argc, char** argv) {
    auto& config = get_experiment_config();

    cxxopts::Options options(argv[0], "A brief description");

    options.add_options()
        ("a,approach", "Approach to use (plain, bws, partitioning)", cxxopts::value<std::string>()->default_value(config.approach))
        ("i,index", "Index type to use", cxxopts::value<std::string>()->default_value(config.index_type))
        ("e,elements", "Number of elements in the index", cxxopts::value<uint64_t>()->default_value(std::to_string(config.num_elements)))
        ("l,lookups", "Size of the lookup dataset", cxxopts::value<uint64_t>()->default_value(std::to_string(config.num_lookups)))
        ("b,blocksize", "Block size", cxxopts::value<unsigned>()->default_value(std::to_string(config.block_size)))
        ("m,maxbits", "Number of radix bits", cxxopts::value<unsigned>()->default_value(std::to_string(config.max_bits)))
        ("z,zipf", "Zipf factor (has no effect when 'lookup_pattern != zipf')", cxxopts::value<double>()->default_value(std::to_string(config.zipf_factor)))
        ("d,dataset", "Index dataset type to generate", cxxopts::value<std::string>()->default_value(tmpl_to_string(config.dataset)))
        ("p,lookup_pattern", "Lookup dataset type to generate", cxxopts::value<std::string>()->default_value(tmpl_to_string(config.lookup_pattern)))
        ("s,sorted_lookups", "Pre-sort the lookup dataset", cxxopts::value<bool>()->default_value(tmpl_to_string(config.sorted_lookups)))
        ("o,output", "File to write results into", cxxopts::value<std::string>()->default_value(config.output_file))
        ("h,help", "Print usage")
    ;

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    // update config state with the parsing results
    config.approach = result["approach"].as<std::string>();
    config.index_type = result["index"].as<std::string>();
    config.num_elements = result["elements"].as<uint64_t>();
    config.num_lookups = result["lookups"].as<uint64_t>();
    config.block_size = result["blocksize"].as<unsigned>();
    config.max_bits = result["maxbits"].as<unsigned>();
    config.zipf_factor = result["zipf"].as<double>();
    config.sorted_lookups = result["sorted_lookups"].as<bool>();
    config.output_file = result["output"].as<std::string>();

    // parse dataset type
    const auto dataset_str = result["dataset"].as<std::string>();
    std::cout << "dataset type: " << dataset_str << std::endl;
    if (dataset_str == "sparse") {
        config.dataset = dataset_type::sparse;
    } else if (dataset_str == "dense") {
        config.dataset = dataset_type::dense;
    } else {
        assert(false);
    }

    // parse lookup pattern type
    const auto lookup_pattern_str = result["lookup_pattern"].as<std::string>();
    if (lookup_pattern_str == "uniform") {
        config.lookup_pattern = lookup_pattern_type::uniform;
    } else if (lookup_pattern_str == "uniform_unique") {
        config.lookup_pattern = lookup_pattern_type::uniform_unique;
    } else if (lookup_pattern_str == "zipf") {
        config.lookup_pattern = lookup_pattern_type::zipf;
    } else {
        assert(false);
    }
}
