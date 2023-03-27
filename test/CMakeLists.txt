
add_executable(test_parser tpch/test_parser.cpp)
set_property(TARGET test_parser PROPERTY CXX_STANDARD 17) # no cuda compiliation units involved; so we can use c++17 here
target_link_libraries(test_parser tpch_common gtest gtest_main pthread)

add_executable(test_utils test_utils.cpp)
set_property(TARGET test_utils PROPERTY CXX_STANDARD 17) # no cuda compiliation units involved; so we can use c++17 here
target_link_libraries(test_utils gtest gtest_main pthread)

add_executable(test_harmonia test_harmonia.cu)
target_link_libraries(test_harmonia cub)
target_compile_options(test_harmonia PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-DNRDC>")

add_executable(test_huge_page_allocator test_huge_page_allocator.cpp)
target_link_libraries(test_huge_page_allocator gtest gtest_main pthread numa)






add_executable(tester
    main.cu
)
set_target_properties(tester PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(tester ${CUDART_LIBRARY} gtest cub) # numa cub radixspline onetbb fast_interconnects)
# additional nvcc flags
target_compile_options(tester PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)
#target_compile_options(index_lookup PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas="-v">)
#target_compile_options(index_lookup PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-DNRDC>")
# debug flag
#target_compile_options(index_lookup PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
target_compile_options(tester PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>")
