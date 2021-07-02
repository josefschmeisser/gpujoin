
add_executable(test_parser tpch/test_parser.cpp) # ../src/tpch/parser.cpp ../src/tpch/common.cpp)
set_property(TARGET test_parser PROPERTY CXX_STANDARD 17) # no cuda compiliation units involved; so we can use c++17 here
target_link_libraries(test_parser tpch_common gtest gtest_main pthread)

add_executable(test_utils test_utils.cpp)
set_property(TARGET test_utils PROPERTY CXX_STANDARD 17) # no cuda compiliation units involved; so we can use c++17 here
target_link_libraries(test_utils gtest gtest_main pthread)
