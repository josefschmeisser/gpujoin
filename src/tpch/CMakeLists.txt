add_library(tpch_common STATIC common.cpp config.cpp ../device_properties.cpp  ../measuring.cpp parser.cpp)
set_property(TARGET tpch_common PROPERTY CXX_STANDARD 17) # no cuda compiliation units involved; so we can use c++17 here
target_link_libraries(tpch_common pthread cub)

set(CPU_TPCH_SRC_FILES cpu_bench.cpp cpu_only.cpp)
add_executable(tpch ${CPU_TPCH_SRC_FILES})
set_property(TARGET tpch PROPERTY CXX_STANDARD 17) # no cuda compiliation units involved; so we can use c++17 here
target_link_libraries(tpch tpch_common ${JEVENTS_LIBRARY_PATH} pthread)
if(ENABLE_PERFBENCHMARKING)
    add_dependencies(tpch jevents)
endif(ENABLE_PERFBENCHMARKING)

add_executable(cuda_tpch_01 tpch_01.cu)
target_link_libraries(cuda_tpch_01 tpch_common numa pthread)

# TODO
#add_executable(cuda_tpch_04 tpch_04.cu)
#set_target_properties(cuda_tpch_04 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(cuda_tpch_04 tpch_common numa pthread radixspline cub fast_interconnects)
#target_compile_options(cuda_tpch_04 PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>")

# index join executable
add_executable(cuda_tpch_14 "")
target_sources(cuda_tpch_14 PRIVATE
    tpch_14_new.cu
    tpch_14_common.cu
    tpch_14_hj.cu
    tpch_14_ij.cu
    tpch_14_ij_partitioning.cu
    ../gpu_radix_partition.cu
    ../harmonia.cu
    ../indexes.cpp
)
set_target_properties(cuda_tpch_14 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cuda_tpch_14 tpch_common numa pthread radixspline cub fast_interconnects)
# additional nvcc flags
target_compile_options(cuda_tpch_14 PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>")
#target_compile_options(cuda_tpch_14 PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-DNRDC>")
# debug flag
#target_compile_options(cuda_tpch_14 PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
target_compile_options(cuda_tpch_14 PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>")
if(false)
    # https://forums.developer.nvidia.com/t/passing-flags-to-nvcc-via-cmake/75768
    # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#ptxas-options-def-load-cache
    target_compile_options(cuda_tpch_14 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        #--generate-line-info
                        #-lineinfo
                        #-Xptxas="-v" # print register usage
                        >)
endif()

if(false)
# split index join executable
add_executable(cuda_tpch_14_split tpch_14_split.cu)
target_link_libraries(cuda_tpch_14_split tpch_common numa pthread radixspline cub)
# additional nvcc flags
endif()
