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
    return (v == lookup_pattern_type::uniform) ? "uniform" : "zipf";
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
        ("dataset", "Index dataset type to generate", cxxopts::value<std::string>()->default_value(tmpl_to_string(config.dataset)))
        ("lookup_pattern", "Lookup dataset type to generate", cxxopts::value<std::string>()->default_value(tmpl_to_string(config.lookup_pattern)))
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
    config.num_elements = result["elements"].as<unsigned>();
    config.num_lookups = result["lookups"].as<unsigned>();
    config.block_size = result["blocksize"].as<unsigned>();
    config.max_bits = result["maxbits"].as<unsigned>();
    config.zipf_factor = result["zipf"].as<double>();

    // parse dataset type
    if (result["dataset"].as<std::string>() == "sparse") {
        config.dataset = dataset_type::sparse;
    } else if (result["dataset"].as<std::string>() == "dense") {
        config.dataset = dataset_type::dense;
    } else {
        assert(false);
    }

    // parse lookup pattern type
    if (result["lookup_pattern"].as<std::string>() == "uniform") {
        config.lookup_pattern = lookup_pattern_type::uniform;
    } else if (result["lookup_pattern"].as<std::string>() == "zipf") {
        config.lookup_pattern = lookup_pattern_type::zipf;
    } else {
        assert(false);
    }
}
