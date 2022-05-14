cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(gpujoin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

set(ENABLE_PERFBENCHMARKING false)

set(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA)

# put predefined cmake projects in their own solution folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# CUDA_ARCHITECTURES was introduced in 3.18, however, we have to support older cmake versions as well
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18.0")
    cmake_policy(SET CMP0104 OLD)
    set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES OFF)
endif()

set(MY_CUDA_TARGET "pascal" CACHE STRING "CUDA target architecture chosen by the user at CMake configure time")
set_property(CACHE MY_CUDA_TARGET PROPERTY STRINGS pascal volta)

if(false)
message(STATUS "current: ${CMAKE_CUDA_FLAGS}")
if("${MY_CUDA_TARGET}" STREQUAL "pascal")
    string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61")
elseif("${MY_CUDA_TARGET}" STREQUAL "volta")
    string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_70,code=compute_70")
endif()
message(STATUS "selected target architecture: ${MY_CUDA_TARGET}")
message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
endif()

#set(CMAKE_CUDA_ARCHITECTURES 60 61 70 CACHE STRING "CUDA architectures")
set(CMAKE_CUDA_ARCHITECTURES 60)
message(STATUS "current: ${CMAKE_CUDA_FLAGS}")

# TODO cmake 3.17: use FindCUDAToolkit()
find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

if(ENABLE_PERFBENCHMARKING)
    include(lib/jevents.cmake)
    add_compile_definitions(PERF_AVAILABLE=true)
endif(ENABLE_PERFBENCHMARKING)

include(lib/RadixSpline.cmake)
include(lib/gtest.cmake)
include(lib/cub.cmake)

include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

#add_executable(btree_lookup src/btree_lookup.cu)
#target_link_libraries(btree_lookup ${CUDART_LIBRARY} numa)

#add_executable(rs_lookup src/rs_lookup.cu)
#target_link_libraries(rs_lookup ${CUDART_LIBRARY} numa radixspline)

add_executable(index_lookup src/index_lookup_new.cu src/measuring.cpp src/device_properties.cpp src/indexes.cu src/index_lookup_config.cpp)
target_link_libraries(index_lookup ${CUDART_LIBRARY} numa cub radixspline)
# additional nvcc flags
if(false)
    # https://forums.developer.nvidia.com/t/passing-flags-to-nvcc-via-cmake/75768
    # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#ptxas-options-def-load-cache
    target_compile_options(index_lookup PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --generate-line-info
                        --use_fast_math
                        --relocatable-device-code=true
                        -Xptxas --def-load-cache=cg
                        -Xptxas -dscm=cg
                        -lineinfo
                        >)
endif()

add_executable(bf_search src/bf_search.cpp)

add_executable(random_access src/random_access.cu)

add_executable(cub_sort src/example_block_radix_sort.cu)
target_link_libraries(cub_sort cub)

add_executable(huge_pages src/huge_pages.cpp)
target_link_libraries(huge_pages numa)

add_subdirectory(src/tpch)
add_subdirectory(test)


add_executable(index_lookup_partitioning
    src/measuring.cpp
    src/device_properties.cpp
    src/index_lookup_config.cpp
    src/index_lookup_partitioning.cu
    src/gpu_radix_partition.cu
)
target_include_directories(index_lookup_partitioning PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/contrib/>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/contrib/numa-gpu/sql-ops/include/>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/contrib/numa-gpu/target/release/build/sql-ops-28848279f8667cf7/out/>
)
target_link_libraries(index_lookup_partitioning ${CUDART_LIBRARY} numa cub radixspline)
# additional nvcc flags
if(true)
    # https://forums.developer.nvidia.com/t/passing-flags-to-nvcc-via-cmake/75768
    # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#ptxas-options-def-load-cache
    target_compile_options(index_lookup_partitioning PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        #--generate-line-info
                        #-lineinfo
                        #-G
                        --default-stream per-thread
                        --expt-relaxed-constexpr
                        >)
endif()
