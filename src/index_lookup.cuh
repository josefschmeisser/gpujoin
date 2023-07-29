#pragma once

#include <memory>
#include <vector>

#include "index_lookup_config.hpp"
#include "device_array.hpp"
#include "indexes.cuh"
#include "measuring.hpp"

struct query_data {
    std::unique_ptr<abstract_index<index_key_t>> index_structure;

    std::vector<index_key_t, host_allocator_t<index_key_t>> indexed, lookup_keys;

    device_array_wrapper<index_key_t> d_indexed;
    device_array_wrapper<index_key_t> d_lookup_keys;
    device_array_wrapper<value_t> d_tids;

    unsigned dataset_max_bits;

    query_data();

    void create_index();

    bool validate_results();
};

struct abstract_approach {
    virtual ~abstract_approach() = default;

    virtual void initialize(query_data& d) = 0;

    virtual void run(query_data& d, measuring::measurement& m) = 0;
};
