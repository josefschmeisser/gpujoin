#include <cassert>
#include <chrono>
#include <iostream>
#include "common.hpp"

#ifdef PERF_AVAILABLE
#include "thirdparty/profile.hpp"
#endif


template<class F>
void measure(F f) {
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto finish = std::chrono::high_resolution_clock::now();
    auto d = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()/1000.;
    std::cout << "Elapsed time: " << d << " ms\n";
}

int main(int argc, char** argv) {
    assert(argc > 1);
    Database db;
    load_tables(db, argv[1]);

#ifdef PERF_AVAILABLE
    PerfEvents e;
    e.timeAndProfile("tpch1", db.lineitem.l_linestatus.size(),
        [&]() {
            query_1(db);
        },
        10, {{"approach", "cpu_only"}, {"threads", std::to_string(1)}});

    e.timeAndProfile("tpch14", db.lineitem.l_linestatus.size(),
        [&]() {
            query_14_lineitem_build(db);
        },
        10, {{"approach", "cpu_only"}, {"threads", std::to_string(1)}});
#endif

    std::cout << "tpch query 1:\n";
    measure([&]() { query_1(db); });

    std::cout << "tpch query 14:\n";
    measure([&]() { query_14_lineitem_build(db); });

    return 0;
}
