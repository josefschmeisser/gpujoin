cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(SimpleConcurrentGPUHashTable LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# put predefined cmake projects in their own solution folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_60,code=sm_60")
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 CACHE STRING "CUDA architectures")

include(lib/jevents.cmake)

include_directories(${CMAKE_SOURCE_DIR}/src)

set(SOURCE_FILES src/gpujoin.cu)
add_executable(gpujoin ${SOURCE_FILES})

add_executable(add_grid src/add_grid.cu)
add_executable(bitonic_sort src/bitonic_sort.cu)

add_executable(btree_lookup src/btree.cpp src/btree_lookup.cu)

add_subdirectory(src/tpch)
