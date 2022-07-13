#include "measuring.hpp"

#include <cassert>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>

#include "utils.hpp"

namespace measuring {

measuring_settings& get_settings() {
    return singleton<measuring_settings>::instance();
};

void record_timestamp(measurement& m) {
    m.timestamps.push_back(std::chrono::high_resolution_clock::now());
}

static auto to_duration_prefix_sum(const measurement& m) {
    assert(m.timestamps.size() > 1);
    std::vector<double> durations;
    const auto start_ts = m.timestamps.front();
    for (size_t i = 1; i < m.timestamps.size(); ++i) {
        const auto stop_ts = m.timestamps[i];
        const double dt = std::chrono::duration_cast<std::chrono::microseconds>(stop_ts - start_ts).count()/1000.;
        durations.push_back(dt);
    }
    return durations;
}

void write_out_measurement(const experiment_description& d, const measurement& m) {
    const auto ts = std::chrono::system_clock::now();
    const uint64_t seconds = std::chrono::duration_cast<std::chrono::seconds>(ts.time_since_epoch()).count();
    const auto duration_prefix_sum = to_duration_prefix_sum(m);

    std::stringstream entry;
    entry << "-\n"
        << "  name: " << d.name << "\n"
        << "  approach: " << d.approach << "\n"
        //<< "  duration: " << m.duration_ms << "\n"
        << "  dt_prefix_sum: " << stringify(duration_prefix_sum.begin(), duration_prefix_sum.end()) << "\n"
        << "  timestamp: " << seconds << "\n";

    for (const auto& misc_pair : d.other) {
        entry << "  " << misc_pair.first << ": " << misc_pair.second << "\n";
    }

    const auto& settings = get_settings();

    if (settings.stdout_only) {
        std::cout << entry.str() << std::endl;
        return;
    }

    // append the entry to the specified file
    const auto file_name = get_settings().dest_file;
    std::ofstream outfile;
    outfile.open(file_name.c_str(), std::ios_base::app);
    outfile << entry.str();
    outfile.close();
}

}; // end namespace measuring
