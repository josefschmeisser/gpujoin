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

void record_timestamp(measurement& m, const std::string& group) {
    using namespace std::chrono;

    const auto end_ts = high_resolution_clock::now();
    // TODO
#if 0
    auto duration = duration_cast<microseconds>(end_ts - m.last_ts);

    if (m.groups.empty()) {
        duration = microseconds::zero();
    }

    {
        std::unique_lock lock(m.m);

        m.dt += duration;

        auto it = m.groups.find(group);
        if (it != m.groups.end()) {
            it->second += duration;
        } else {
            m.groups.emplace(group, duration);
        }
    }

    //m.last_ts = high_resolution_clock::now();
#endif
}

/*
static auto to_duration_prefix_sum(const measurement& m) {
    assert(m.timestamps.size() > 1);
    std::vector<std::pair<std::string, double>> durations;
    const auto start_ts = m.timestamps.front().second;
    for (size_t i = 1; i < m.timestamps.size(); ++i) {
        const auto tuple = m.timestamps[i];
        const auto stop_ts = tuple.second;
        const double dt = std::chrono::duration_cast<std::chrono::microseconds>(stop_ts - start_ts).count()/1000.;
        durations.push_back(std::make_pair(tuple.first, dt));
    }
    return durations;
}
*/

template<class Iter>
static std::string serialize_groups(const Iter begin, const Iter end) {
    using namespace std::chrono;

    std::stringstream s;
    s << "[";
    for (auto it = begin; it != end; ++it) {
        if (it != begin) {
            s << ", ";
        }
        const int64_t dt_ms = duration<double, milliseconds::period>(it->second).count(); // TODO to ms
        s << "{ group: " << it->first << ", value: " << dt_ms << " }"; 
    }
    s << "]";
    return s.str();
}

void write_out_measurement(const experiment_description& d, const measurement& m) {
    using namespace std::chrono;

    const auto ts = system_clock::now();
    //const uint64_t seconds = duration_cast<seconds>(ts.time_since_epoch()).count();
    const uint64_t seconds = duration_cast<std::chrono::seconds>(ts.time_since_epoch()).count();

    std::stringstream entry;
    entry << "-\n"
        << "  name: " << d.name << "\n"
        << "  approach: " << d.approach << "\n"
        << "  dt: " << duration_cast<milliseconds>(m.dt).count() << "\n" // TODO to ms
        << "  dt_breakdown: " << serialize_groups(m.groups.begin(), m.groups.end()) << "\n"
        << "  timestamp: !int64 " << seconds << "\n";

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
