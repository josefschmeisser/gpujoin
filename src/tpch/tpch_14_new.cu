#include <iostream>

#include "config.hpp"
#include "measuring.hpp"
#include "tpch_14_common.cuh"

int main(int argc, char** argv) {
    parse_options(argc, argv);
    const auto& config = get_experiment_config();

    // set-up the measuring utility
    auto& measuring_config = measuring::get_settings();
    measuring_config.dest_file = "tpch_14_results.yml";
    measuring_config.repetitions = 10;
    // TODO
    //const auto experiment_desc = create_experiment_description();

    execute_approach("streamed");
    return 0;
}
