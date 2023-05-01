#pragma once

#include "index_lookup.cuh"
#include "measuring.hpp"

template<class IndexType>
struct partitioning_approach {
    void operator()(query_data& d, measuring::measurement& m);
};
