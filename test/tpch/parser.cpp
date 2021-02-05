
// TODO

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <lineitem.tbl>" << std::endl;
        return 1;
    }

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
    auto result = parse<lineitem_tuple>(argv[1]);

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
    load_tables(db, argv[2]);
    sort_relation(db.lineitem);

    printf("comparing...\n");
    auto& my_l_orderkey = *std::get<0>(result);
    for (size_t i = 0; i < db.lineitem.l_orderkey.size(); ++i) {
        if (my_l_orderkey[i] != db.lineitem.l_orderkey[i]) {
            printf("for i == %lu (my_l_orderkey[i] == %u) != (db.lineitem.l_orderkey[i] == %u)\n", i, my_l_orderkey[i], db.lineitem.l_orderkey[i]);
            throw 0;
        }
    }

    return 0;
}
