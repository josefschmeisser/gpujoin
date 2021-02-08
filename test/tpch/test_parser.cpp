#include <iostream>

#include <gtest/gtest.h>

#include "tpch/common.hpp"
#include "tpch/parser.hpp"

char* g_tpch_path;

int main(int argc, char** argv) {
    if (argc < 1) {
        std::cout << "Usage: " << argv[0] << " <tpch path>" << std::endl;
        return 1;
    }
    g_tpch_path = argv[1];

    ::testing::InitGoogleTest(&argc, argv);

    for (int i = 1; i < argc; ++i) {
        printf("arg %2d = %s\n", i, argv[i]);
    }

    return RUN_ALL_TESTS();
}

TEST(test_parser, lineite_test) {
    std::string lineitem_file = std::string(g_tpch_path) + "lineitem.tbl";
    using lineitem_tuple = std::tuple<
        uint32_t, // l_orderkey
        uint32_t,
        uint32_t,
        uint32_t,
        numeric<15, 2>, // l_quantity
        numeric<15, 2>,
        numeric<15, 2>,
        numeric<15, 2>,
        char, // l_returnflag
        char,
        date, // l_shipdate
        date,
        date,
        std::array<char, 25>, // l_shipinstruct
        std::array<char, 10>, // l_shipmode
        std::array<char, 44>  // l_comment
        >;
    auto result = parse<lineitem_tuple>(lineitem_file);

    printf("sorting relation...\n");
    do_sort(result);

/*
    ofstream myfile;
    myfile.open ("example.txt");
    write_out(result, myfile);
    myfile.close();

    return 0;
*/

    printf("loading ground truth...\n");

    Database db;
    load_tables_with_aria_parser(db, g_tpch_path);
    sort_relation(db.lineitem);

    printf("comparing...\n");
    auto& my_l_orderkey = *std::get<0>(result);
    for (size_t i = 0; i < db.lineitem.l_orderkey.size(); ++i) {
        ASSERT_EQ(my_l_orderkey[i], db.lineitem.l_orderkey[i]);/*
        if (my_l_orderkey[i] != db.lineitem.l_orderkey[i]) {
            printf("for i == %lu (my_l_orderkey[i] == %u) != (db.lineitem.l_orderkey[i] == %u)\n", i, my_l_orderkey[i], db.lineitem.l_orderkey[i]);
            throw 0;
        }*/
    }
}
