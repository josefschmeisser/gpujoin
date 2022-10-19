# index search executable
add_executable(search "")
target_sources(search PRIVATE
    tpch_14_new.cu
    tpch_14_common.cu
    tpch_14_hj.cu
    tpch_14_ij.cu
    tpch_14_ij_partitioning.cu
    ../gpu_radix_partition.cu
    ../harmonia.cu
    ../indexes.cu
)
set_target_properties(search PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#target_include_directories(search PUBLIC
#    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/contrib/>
#)
target_link_libraries(search tpch_common numa pthread radixspline cub)
# additional nvcc flags
target_compile_options(search PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>")
#target_compile_options(search PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-DNRDC>")
# debug flag
#target_compile_options(search PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
target_compile_options(search PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>")
if(false)
    # https://forums.developer.nvidia.com/t/passing-flags-to-nvcc-via-cmake/75768
    # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#ptxas-options-def-load-cache
    target_compile_options(search PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        #--generate-line-info
                        #-lineinfo
                        #-Xptxas="-v" # print register usage
                        >)
endif()
