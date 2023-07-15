#pragma once

#include <memory>

#include "index_lookup.cuh"
#include "measuring.hpp"

template<class IndexType>
struct partitioning_approach : abstract_approach {
    partitioning_approach();

    ~partitioning_approach();

    void initialize(query_data& d) override;

    void run(query_data& d, measuring::measurement& m) override;

private:
    struct impl;
    std::unique_ptr<impl> _p_impl;
};
