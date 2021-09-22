#include <algorithm>
#include <functional>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "utils.hpp"

TEST(test_utils, apply_permutation) {
    std::vector<int> vec { 3, 5, 2, 8, 1, 4 };
    auto per = compute_permutation(vec.begin(), vec.end(), std::less<>{});
    apply_permutation(per, vec);
    auto pos = std::adjacent_find(vec.begin(), vec.end(), std::greater<>{});
    ASSERT_EQ(pos, vec.end());
}
