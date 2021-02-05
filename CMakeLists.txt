cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(gpujoin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

set(ENABLE_PERFBENCHMARKING false)

set(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA)

# put predefined cmake projects in their own solution folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

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

include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

add_executable(btree_monolithic src/btree_monolithic.cu)
target_link_libraries(btree_monolithic ${CUDART_LIBRARY} numa)

add_executable(rs_lookup src/rs_lookup.cu)
add_dependencies(rs_lookup RadixSpline)
target_include_directories(rs_lookup PRIVATE $<BUILD_INTERFACE:${RADIXSPLINE_INCLUDE_DIR}>)

add_subdirectory(src/tpch)

add_executable(bf_search src/bf_search.cpp)

#add_executable(parser src/csv.cpp src/tpch/common.cpp)
#target_link_libraries(parser pthread)
