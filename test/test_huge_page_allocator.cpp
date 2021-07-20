#include <algorithm>
#include <functional>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "utils.hpp"
#include "huge_page_allocator.hpp"

TEST(test_huge_page_allocator, allocate) {
    huge_page_allocator<int> a(0, huge_page_allocator<int>::huge_2mb);
    int* ptr;
    EXPECT_NO_THROW(ptr = a.allocate(1));
    EXPECT_NO_THROW(a.deallocate(ptr, 1));
}

TEST(test_huge_page_allocator, memset) {
    huge_page_allocator<int> a(0, huge_page_allocator<int>::huge_2mb);
    int* ptr;
    EXPECT_NO_THROW(ptr = a.allocate(1));
    std::memset(ptr, 0xff, sizeof(int));
    ASSERT_EQ(*ptr, 0xffffffff);
    EXPECT_NO_THROW(a.deallocate(ptr, 1));
}
