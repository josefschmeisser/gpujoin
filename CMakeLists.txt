cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(gpujoin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

set(ENABLE_PERFBENCHMARKING false)

set(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA)

# put predefined cmake projects in their own solution folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# CUDA_ARCHITECTURES was introduced in 3.18, however, we have to support older cmake versions as well
cmake_policy(SET CMP0104 OLD)
set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES OFF)

set(MY_CUDA_TARGET "pascal" CACHE STRING "CUDA target architecture chosen by the user at CMake configure time")
set_property(CACHE MY_CUDA_TARGET PROPERTY STRINGS pascal volta)

if("${MY_CUDA_TARGET}" STREQUAL "pascal")
    string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61")
elseif("${MY_CUDA_TARGET}" STREQUAL "volta")
    string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_70,code=compute_70")
endif()
message(STATUS "selected target architecture: ${MY_CUDA_TARGET}")

#set(CMAKE_CUDA_ARCHITECTURES 60 61 70 CACHE STRING "CUDA architectures")
#set(CMAKE_CUDA_ARCHITECTURES 60 61 70)

# TODO cmake 3.17: use FindCUDAToolkit()
find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

if(ENABLE_PERFBENCHMARKING)
    include(lib/jevents.cmake)
    add_compile_definitions(PERF_AVAILABLE=true)
endif(ENABLE_PERFBENCHMARKING)

include(lib/RadixSpline.cmake)
include(lib/gtest.cmake)
include(lib/cub.cmake)

message(STATUS "cub include dir: ${CUB_INCLUDE_DIR}")

include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

add_executable(btree_lookup src/btree_lookup.cu)
target_link_libraries(btree_lookup ${CUDART_LIBRARY} numa)

#add_executable(rs_lookup src/rs_lookup.cu)
#add_dependencies(rs_lookup RadixSpline)
#target_link_libraries(rs_lookup ${CUDART_LIBRARY} numa)
#target_include_directories(rs_lookup PRIVATE $<BUILD_INTERFACE:${RADIXSPLINE_INCLUDE_DIR}>)

add_executable(index_lookup src/index_lookup.cu)
add_dependencies(index_lookup RadixSpline cub)
target_link_libraries(index_lookup ${CUDART_LIBRARY} numa)
target_include_directories(index_lookup PRIVATE "$<BUILD_INTERFACE:${RADIXSPLINE_INCLUDE_DIR};${CUB_INCLUDE_DIR}>")
# https://forums.developer.nvidia.com/t/passing-flags-to-nvcc-via-cmake/75768
# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#ptxas-options-def-load-cache
target_compile_options(index_lookup PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#                       --generate-line-info
#                       --use_fast_math
#                       --relocatable-device-code=true
#                        -Xptxas --def-load-cache=cg
#                        -Xptxas -dscm=cg
#                        -lineinfo
                       >)

add_subdirectory(src/tpch) # TODO
add_subdirectory(test)

add_executable(bf_search src/bf_search.cpp)

add_executable(random_access src/random_access.cu)

add_executable(cub_sort src/example_block_radix_sort.cu)

add_executable(huge_pages src/huge_pages.cpp)
target_link_libraries(huge_pages numa)
