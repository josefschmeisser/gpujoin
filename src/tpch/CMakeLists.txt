set(SOURCE_FILES cpu.cpp common.cpp)

#include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(tpch ${SOURCE_FILES})

add_executable(tpch_gpu gpu.cu common.cpp)
