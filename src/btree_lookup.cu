#include "btree.hpp"

#include <iostream>

using namespace std;




int main() {


    auto tree = btree::construct_dense(1e6, 0.7);
    for (unsigned i = 0; i < 1e6; ++i) {
        printf("lookup %d\n", i);
        btree::payload_t value;
        bool found = btree::lookup(tree, i, value);
        if (!found) throw 0;
    }
    
    return 0;
}
