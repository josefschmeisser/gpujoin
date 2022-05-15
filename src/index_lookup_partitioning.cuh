#pragma once

#include "index_lookup.cuh"

template<class IndexType>
struct partitioning_approach {
    void operator()(query_data& d);
};
