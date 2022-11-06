#include "indexes.hpp"

#include <stdexcept>

index_type_enum parse_index_type(const std::string& index_name) {
    if (index_name == "btree") {
        return index_type_enum::btree;
    } else if (index_name == "harmonia") {
        return index_type_enum::harmonia;
    } else if (index_name == "binary_search") {
        return index_type_enum::binary_search;
    } else if (index_name == "radix_spline") {
        return index_type_enum::radix_spline;
    } else if (index_name == "no_op") {
        return index_type_enum::no_op;
    } else {
        throw std::runtime_error("unknown index type");
    }
}
