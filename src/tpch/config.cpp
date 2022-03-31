#include "config.hpp"

#include <cassert>
#include <iostream>
#include <string>

#include "utils.hpp"
#include "thirdparty/cxxopts.hpp"

experiment_config& get_experiment_config() {
    return singleton<experiment_config>::instance();
}

void parse_options(int argc, char** argv) {
    auto& config = get_experiment_config();

    cxxopts::Options options(argv[0], "TPC-H query 14 runner");

    options.add_options()
        // TODO
        ("i,index", "Index type to use", cxxopts::value<std::string>()->default_value(config.index_type))
        ("h,help", "Print usage")
    ;

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    config.index_type = result["index"].as<std::string>();

#if 0
    // update config state with the parsing results
    config.num_elements = result["elements"].as<unsigned>();
    config.num_lookups = result["lookups"].as<unsigned>();
    config.max_bits = result["maxbits"].as<unsigned>();
    config.zipf_factor = result["zipf"].as<double>();
    config.partitial_sorting = result["partial"].as<bool>();

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
#endif
}
