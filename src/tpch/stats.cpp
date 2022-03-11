#include <cassert>
#include <cstddef>
#include <iostream>
#include "common.hpp"

struct histrogram {
    std::vector<unsigned> bins;

    histrogram(unsigned bin_cnt) {
    };

};

void query_14_stats(Database& db) {
    int64_t sum1 = 0;
    int64_t sum2 = 0;

    constexpr auto lower_shipdate = to_julian_day(1, 9, 1995); // 1995-09-01
    constexpr auto upper_shipdate = to_julian_day(1, 10, 1995); // 1995-10-01

    auto& lineitem = db.lineitem;

    size_t count = 0;
    double avg = 0.;

    // aggregation loop
    for (size_t i = 0; i < lineitem.l_partkey.size(); ++i) {
        if (lineitem.l_shipdate[i].raw < lower_shipdate ||
            lineitem.l_shipdate[i].raw >= upper_shipdate) {
            continue;
        }

        avg += lineitem.l_partkey;
        count++;
    }
    avg /= static_cast<double>(count);


}

int main(int argc, char** argv) {
    assert(argc > 1);
    Database db;
    load_tables(db, argv[1]);


    return 0;
}
