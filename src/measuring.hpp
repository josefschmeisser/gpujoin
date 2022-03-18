#pragma once

#include <chrono>
#include <string>
#include <tuple>
#include <vector>

namespace measuring {

struct measuring_settings {
    std::string dest_file;
    unsigned repetitions;
};

struct experiment_description {
    std::string name;
    std::string approach;
    std::vector<std::pair<std::string, std::string>> other;
};

struct measurement {
/*
    std::string experiment;
    std::string approach;*/
    std::string device; // TODO move
/*
    std::string index_type;
    std::string index_config;
    std::string host_allocator;
    std::string index_allocator;
    std::string query_allocator;*/
    //std::string other;
    std::string short_commit_hash;
    double duration_ms;
    uint32_t timestamp;
    uint64_t count;
};

measuring_settings& get_settings();

void write_out_measurement(const experiment_description& d, const measurement& m);

template<class Func>
void measure(const experiment_description& d, Func func) {
    printf("repetitions: %u\n", get_settings().repetitions);
    for (unsigned i = 0; i < get_settings().repetitions; ++i) {
        const auto start_ts = std::chrono::high_resolution_clock::now();
        func();
        const auto stop_ts = std::chrono::high_resolution_clock::now();
        const double rt = std::chrono::duration_cast<std::chrono::microseconds>(stop_ts - start_ts).count()/1000.;

        measurement m;
        m.duration_ms = rt;

        write_out_measurement(d, m);
    }
};

}; // end namespace measuring
