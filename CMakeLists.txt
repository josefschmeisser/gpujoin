cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(SimpleConcurrentGPUHashTable LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# put predefined cmake projects in their own solution folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_60,code=sm_60")
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 CACHE STRING "CUDA architectures")

# TODO cmake 3.17: use FindCUDAToolkit()
find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

include(lib/jevents.cmake)
include(lib/RadixSpline.cmake)

include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${RADIXSPLINE_INCLUDE_DIR}
)

add_executable(btree_lookup src/btree.cpp src/btree_lookup.cu)
target_include_directories(
    btree_lookup
    PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_libraries(
    btree_lookup
    ${CUDART_LIBRARY}
)

add_executable(rs_lookup src/rs_lookup.cu)

add_subdirectory(src/tpch)
