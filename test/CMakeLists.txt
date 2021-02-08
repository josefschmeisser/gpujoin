
add_executable(test_parser tpch/test_parser.cpp ../src/tpch/parser.cpp ../src/tpch/common.cpp)
target_link_libraries(test_parser gtest gtest_main pthread)
