
add_executable(test_parser tpch/test_parser.cpp) # ../src/tpch/parser.cpp ../src/tpch/common.cpp)
target_link_libraries(test_parser tpch_common gtest gtest_main pthread)

add_executable(test_utils test_utils.cpp)
target_link_libraries(test_utils gtest gtest_main pthread)
