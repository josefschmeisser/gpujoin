#pragma once

#include <string>

enum class index_type_enum : unsigned { btree, harmonia, lower_bound, radix_spline, no_op };

index_type_enum parse_index_type(const std::string& index_name);
