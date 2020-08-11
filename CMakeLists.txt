cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(SimpleConcurrentGPUHashTable LANGUAGES CXX CUDA)

# put predefined cmake projects in their own solution folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_60,code=sm_60")

set(SOURCE_FILES src/gpujoin.cu)

include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(gpujoin ${SOURCE_FILES})

