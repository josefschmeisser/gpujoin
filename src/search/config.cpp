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
        ("positional", "Input", cxxopts::value<std::vector<std::string>>())
        ("a,approach", "Approach to use (hj, ij_plain, ij_partitioning)", cxxopts::value<std::string>()->default_value(config.approach))
        ("b,blocksize", "Block size", cxxopts::value<unsigned>()->default_value(std::to_string(config.block_size)))
        ("d,destfile", "Output file", cxxopts::value<std::string>()->default_value(config.dest_file))
        ("i,index", "Index type to use", cxxopts::value<std::string>()->default_value(config.index_type))
        ("h,help", "Print usage")
    ;

    options.parse_positional("positional");
    options.positional_help("<input>");
    //options.show_positional_help();

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    auto& positional = result["positional"].as<std::vector<std::string>>();
    if (positional.size() != 1) {
        std::cerr << "No database given" << std::endl;
        exit(1);
    }

    config.approach = result["approach"].as<std::string>();
    config.block_size = result["blocksize"].as<unsigned>();
    config.dest_file = result["destfile"].as<std::string>();
    config.index_type = result["index"].as<std::string>();
    config.db_path = positional.front();
}
