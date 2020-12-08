set(SOURCE_FILES cpu_bench.cpp common.cpp cpu_only.cpp)

#include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(tpch ${SOURCE_FILES})
target_include_directories(tpch PRIVATE $<BUILD_INTERFACE:${JEVENTS_INCLUDE_DIR}>)
target_link_libraries(tpch ${JEVENTS_LIBRARY_PATH})

if(ENABLE_PERFBENCHMARKING)
    add_dependencies(tpch jevents)
endif(ENABLE_PERFBENCHMARKING)

add_executable(cuda_tpch_1 tpch_1.cu common.cpp)

add_executable(cuda_tpch_14 tpch_14.cu common.cpp)
