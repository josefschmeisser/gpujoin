message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

add_library(tpch_common STATIC common.cpp parser.cpp)
set_property(TARGET tpch_common PROPERTY CXX_STANDARD 17) # no cuda compiliation units involved; so we can use c++17 here
target_link_libraries(tpch_common pthread)

set(CPU_TPCH_SRC_FILES cpu_bench.cpp cpu_only.cpp)
add_executable(tpch ${CPU_TPCH_SRC_FILES})
set_property(TARGET tpch PROPERTY CXX_STANDARD 17) # no cuda compiliation units involved; so we can use c++17 here
#target_include_directories(tpch PRIVATE $<BUILD_INTERFACE:${JEVENTS_INCLUDE_DIR}>)
target_link_libraries(tpch tpch_common ${JEVENTS_LIBRARY_PATH} pthread)
if(ENABLE_PERFBENCHMARKING)
    add_dependencies(tpch jevents)
endif(ENABLE_PERFBENCHMARKING)

add_executable(cuda_tpch_1 tpch_1.cu)
target_link_libraries(cuda_tpch_1 tpch_common numa pthread)

#list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70)
#string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70")
#add_executable(cuda_tpch_14 ../btree.cu tpch_14.cu common.cpp)
#cuda_add_executable(cuda_tpch_14 ../btree.cu tpch_14.cu common.cpp)
#target_compile_options(cuda_tpch_14 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-dc>)
#target_link_libraries(cuda_tpch_14 numa)

# index join executable
add_executable(cuda_tpch_14 tpch_14.cu)
target_link_libraries(cuda_tpch_14 tpch_common numa pthread radixspline cub)
#target_include_directories(cuda_tpch_14 PRIVATE "$<BUILD_INTERFACE:${RADIXSPLINE_INCLUDE_DIR};${CUB_INCLUDE_DIR}>")

# hash join executable
add_executable(cuda_hj_tpch_14 tpch_14.cu)
target_compile_definitions(cuda_hj_tpch_14 PUBLIC USE_HJ)
target_link_libraries(cuda_hj_tpch_14 tpch_common numa pthread radixspline cub)
#target_include_directories(cuda_hj_tpch_14 PRIVATE "$<BUILD_INTERFACE:${RADIXSPLINE_INCLUDE_DIR};${CUB_INCLUDE_DIR}>")
