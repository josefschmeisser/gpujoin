add_library(tpch_common STATIC common.cpp parser.cpp)
set_property(TARGET tpch_common PROPERTY CXX_STANDARD 17) # no cuda compiliation units involved; so we can use c++17 here
target_link_libraries(tpch_common pthread)

set(CPU_TPCH_SRC_FILES cpu_bench.cpp cpu_only.cpp)
add_executable(tpch ${CPU_TPCH_SRC_FILES})
set_property(TARGET tpch PROPERTY CXX_STANDARD 17) # no cuda compiliation units involved; so we can use c++17 here
target_link_libraries(tpch tpch_common ${JEVENTS_LIBRARY_PATH} pthread)
if(ENABLE_PERFBENCHMARKING)
    add_dependencies(tpch jevents)
endif(ENABLE_PERFBENCHMARKING)

add_executable(cuda_tpch_1 tpch_1.cu)
target_link_libraries(cuda_tpch_1 tpch_common numa pthread)

# index join executable
add_executable(cuda_tpch_14 tpch_14.cu)
target_link_libraries(cuda_tpch_14 tpch_common numa pthread radixspline cub)

# hash join executable
add_executable(cuda_hj_tpch_14 tpch_14.cu)
target_compile_definitions(cuda_hj_tpch_14 PUBLIC USE_HJ)
target_link_libraries(cuda_hj_tpch_14 tpch_common numa pthread radixspline cub)
