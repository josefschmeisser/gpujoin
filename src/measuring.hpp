#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>
#include <iostream>

namespace measuring {

static unsigned warm_up_rounds = 3;

struct measuring_settings {
    std::string dest_file;
    unsigned repetitions;
    bool stdout_only = false;
};

struct experiment_description {
    std::string name;
    std::string approach;
    std::vector<std::pair<std::string, std::string>> other;
};

struct measurement {
    std::vector<decltype(std::chrono::high_resolution_clock::now())> timestamps;
    std::string str_rep;
};

measuring_settings& get_settings();

void write_out_measurement(const experiment_description& d, const measurement& m);

void record_timestamp(measurement& m);

struct nop_validator {
    bool operator() () { return true; }
};

template<class Func, class Validator = nop_validator>
void measure(const experiment_description& d, Func func, Validator validator = nop_validator{}) {
    for (unsigned i = 0; i < warm_up_rounds; ++i) {
        measurement m;
        func(m);
    }
    for (unsigned i = 0; i < get_settings().repetitions; ++i) {
        measurement m;

        record_timestamp(m);
        func(m);
        record_timestamp(m);

        if (validator()) {
            write_out_measurement(d, m);
        } else {
            std::cerr << "invalid measurement" << std::endl;
        }
    }
};

template<class Func>
void measure_only(Func func) {
    for (unsigned i = 0; i < warm_up_rounds; ++i) {
        func();
    }
    for (unsigned i = 0; i < get_settings().repetitions; ++i) {
        const auto start_ts = std::chrono::high_resolution_clock::now();
        func();
        const auto stop_ts = std::chrono::high_resolution_clock::now();
        const double rt = std::chrono::duration_cast<std::chrono::microseconds>(stop_ts - start_ts).count()/1000.;
        std::cout << "elapsed time: " << rt << " ms" << std::endl;
    }
};

}; // end namespace measuring
