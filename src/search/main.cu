#include <iostream>

#include "config.hpp"
#include "measuring.hpp"
#include "join_common.cuh"

int main(int argc, char** argv) {
    parse_options(argc, argv);
    const auto& config = get_experiment_config();

    // set-up the measuring utility
    auto& measuring_config = measuring::get_settings();

    if (!config.dest_file.empty()) {
        measuring_config.dest_file = config.dest_file;
        measuring_config.repetitions = 3;
    } else {
        measuring_config.repetitions = 1;
        measuring_config.stdout_only = true;
    }
    // TODO
    //const auto experiment_desc = create_experiment_description();

    execute_approach(config.approach);

    return 0;
}
