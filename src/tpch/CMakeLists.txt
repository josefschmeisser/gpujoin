add_library(tpch_common STATIC common.cpp config.cpp ../device_properties.cpp  ../measuring.cpp parser.cpp)
set_property(TARGET tpch_common PROPERTY CXX_STANDARD 17) # no cuda compiliation units involved; so we can use c++17 here
target_link_libraries(tpch_common pthread)

set(CPU_TPCH_SRC_FILES cpu_bench.cpp cpu_only.cpp)
add_executable(tpch ${CPU_TPCH_SRC_FILES})
set_property(TARGET tpch PROPERTY CXX_STANDARD 17) # no cuda compiliation units involved; so we can use c++17 here
target_link_libraries(tpch tpch_common ${JEVENTS_LIBRARY_PATH} pthread)
if(ENABLE_PERFBENCHMARKING)
    add_dependencies(tpch jevents)
endif(ENABLE_PERFBENCHMARKING)

add_executable(cuda_tpch_1 tpch_1.cu)
target_link_libraries(cuda_tpch_1 tpch_common numa pthread)

# index join executable
add_executable(cuda_tpch_14 tpch_14_new.cu tpch_14_common.cu tpch_14_hj.cu tpch_14_streamed_ij.cu) # ../cuda_utils.cu)
target_link_libraries(cuda_tpch_14 tpch_common numa pthread radixspline cub)
# additional nvcc flags
if(true)
    # https://forums.developer.nvidia.com/t/passing-flags-to-nvcc-via-cmake/75768
    # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#ptxas-options-def-load-cache
    target_compile_options(cuda_tpch_14 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        #--generate-line-info
                        -lineinfo
                        -Xptxas="-v" # print register usage
                        >)
endif()

if(false)
# split index join executable
add_executable(cuda_tpch_14_split tpch_14_split.cu)
target_link_libraries(cuda_tpch_14_split tpch_common numa pthread radixspline cub)
# additional nvcc flags
if(true)
    # https://forums.developer.nvidia.com/t/passing-flags-to-nvcc-via-cmake/75768
    # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#ptxas-options-def-load-cache
    target_compile_options(cuda_tpch_14_split PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        #--generate-line-info
                        -lineinfo
                        #-G
                        #-Xptxas="-v" # print register usage
                        >)
endif()

# hash join executable
add_executable(cuda_hj_tpch_14 tpch_14.cu)
target_compile_definitions(cuda_hj_tpch_14 PUBLIC USE_HJ)
target_link_libraries(cuda_hj_tpch_14 tpch_common numa pthread radixspline cub)
endif()
