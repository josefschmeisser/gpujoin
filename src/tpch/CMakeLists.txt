set(SOURCE_FILES cpu_bench.cpp common.cpp cpu_only.cpp)

#include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(tpch ${SOURCE_FILES})
target_include_directories(tpch PRIVATE $<BUILD_INTERFACE:${JEVENTS_INCLUDE_DIR}>)
target_link_libraries(tpch ${JEVENTS_LIBRARY_PATH})

if(ENABLE_PERFBENCHMARKING)
    add_dependencies(tpch jevents)
endif(ENABLE_PERFBENCHMARKING)

add_executable(cuda_tpch_1 tpch_1.cu common.cpp)
target_link_libraries(cuda_tpch_1 numa)

list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70)
#string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70")

#add_executable(cuda_tpch_14 ../btree.cu tpch_14.cu common.cpp)
cuda_add_executable(cuda_tpch_14 ../btree.cu tpch_14.cu common.cpp)
#target_compile_options(cuda_tpch_14 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-dc>)
#target_compile_options(cuda_tpch_14 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70>)
target_link_libraries(cuda_tpch_14 numa)
