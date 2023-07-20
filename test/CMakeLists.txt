add_executable(test_parser tpch/test_parser.cpp)
set_property(TARGET test_parser PROPERTY CXX_STANDARD 17) # no cuda compiliation units involved; so we can use c++17 here
target_link_libraries(test_parser tpch_common gtest gtest_main pthread)

add_executable(tester
    main.cu
    test_harmonia.cu
    test_huge_page_allocator.cpp
    test_utils.cpp
    ../src/harmonia.cu
)
set_target_properties(tester PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(tester ${CUDART_LIBRARY} gtest cub pthread numa)
# additional nvcc flags
target_compile_options(tester PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)
#target_compile_options(index_lookup PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas="-v">)
#target_compile_options(index_lookup PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-DNRDC>")
# debug flag
#target_compile_options(index_lookup PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
target_compile_options(tester PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>")
