cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(gpujoin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(ENABLE_PERFBENCHMARKING false)

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

# Set CUDA target architectures
#set(CMAKE_CUDA_ARCHITECTURES 60 61 70 CACHE STRING "CUDA architectures")
#set(CMAKE_CUDA_ARCHITECTURES 60)
message(STATUS "CUDA target architectures: ${CMAKE_CUDA_FLAGS}")

# TODO cmake 3.17: use FindCUDAToolkit()
find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

if(ENABLE_PERFBENCHMARKING)
    include(lib/jevents.cmake)
    add_compile_definitions(PERF_AVAILABLE=true)
endif(ENABLE_PERFBENCHMARKING)

include(lib/RadixSpline.cmake)
include(lib/gtest.cmake)
include(lib/cub.cmake)
include(lib/oneTBB.cmake)
include(lib/fast-interconnects.cmake)
include(lib/cuCollections.cmake)

# CUB does not use NDEBUG but rather DEBUG and _DEBUG, so we define the latter here
add_compile_options("$<$<CONFIG:Debug>:-D_DEBUG>")

include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

#add_executable(btree_lookup src/btree_lookup.cu)
#target_link_libraries(btree_lookup ${CUDART_LIBRARY} numa)

#add_executable(rs_lookup src/rs_lookup.cu)
#target_link_libraries(rs_lookup ${CUDART_LIBRARY} numa radixspline)

add_executable(index_lookup
    src/generic_hj.cu
    src/index_lookup_new.cu
    src/index_lookup_partitioning.cu
    src/measuring.cpp
    src/device_properties.cpp
    src/harmonia.cu
    src/indexes.cpp
    src/index_lookup_config.cpp
    src/gpu_radix_partition.cu
)
set_target_properties(index_lookup PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(index_lookup ${CUDART_LIBRARY} numa cub radixspline onetbb fast_interconnects cuCollections)
# additional nvcc flags
target_compile_options(index_lookup PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)
#target_compile_options(index_lookup PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas="-v">)
#target_compile_options(index_lookup PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-DNRDC>")
# debug flag
#target_compile_options(index_lookup PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
target_compile_options(index_lookup PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>")
if(false)
    # https://forums.developer.nvidia.com/t/passing-flags-to-nvcc-via-cmake/75768
    # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#ptxas-options-def-load-cache
    target_compile_options(index_lookup PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        #--generate-line-info
                        #--use_fast_math
                        #-Xptxas --def-load-cache=cg
                        #-Xptxas -dscm=cg
                        >)
endif()


# FIXME
if(false)
add_executable(index_lookup_partitioning
    src/index_lookup_partitioning_old.cu
    src/measuring.cpp
    src/device_properties.cpp
    #src/harmonia.cu
    src/indexes.cpp
    src/index_lookup_config.cpp
    src/gpu_radix_partition.cu
)
target_link_libraries(index_lookup_partitioning ${CUDART_LIBRARY} numa cub radixspline onetbb fast_interconnects)
# additional nvcc flags
target_compile_options(index_lookup_partitioning PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-DNRDC>")
endif()

add_executable(bf_search src/bf_search.cpp)

add_executable(random_access src/random_access.cu)

add_executable(cub_sort src/example_block_radix_sort.cu)
target_link_libraries(cub_sort cub)

add_executable(huge_pages src/huge_pages.cpp)
target_link_libraries(huge_pages numa)

add_subdirectory(src/tpch)
add_subdirectory(test)
