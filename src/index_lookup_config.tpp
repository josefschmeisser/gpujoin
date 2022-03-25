#pragma once

#include "utils.hpp"

template<>
std::string tmpl_to_string(const dataset_type& v);

template<>
std::string tmpl_to_string(const lookup_pattern_type& v);
