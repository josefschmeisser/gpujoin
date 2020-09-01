#include <cassert>
#include "common.hpp"

int main(int argc, char** argv) {
    assert(argc > 1);
    Database db;
    load_tables(db, argv[1]);
    query_14(db);
    return 0;
}
