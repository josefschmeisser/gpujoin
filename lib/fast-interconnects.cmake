include(ExternalProject)
find_package(Git REQUIRED)
#find_package(Rust REQUIRED) # TODO

find_program(CARGO "cargo")
IF(NOT CARGO)
    message(FATAL_ERROR "cargo not found!")
ENDIF()

set(FAST_INTERCONNECTS_PREFIX "fast-interconnects")

message(STATUS "nvcc: ${CMAKE_CUDA_COMPILER} cuda root: ${CUDA_TOOLKIT_ROOT_DIR}")

get_filename_component(_NVCC_DIR ${CMAKE_CUDA_COMPILER} PATH)
message(STATUS "nvcc dir: ${_NVCC_DIR}")

# Get FAST_INTERCONNECTS
ExternalProject_Add(
    fast_interconnects_src
    PREFIX ${FAST_INTERCONNECTS_PREFIX}
    GIT_REPOSITORY "git@github.com:TU-Berlin-DIMA/fast-interconnects.git"
    GIT_TAG "84c181a"
    TIMEOUT 10
    CONFIGURE_COMMAND ""
    #BUILD_COMMAND ${CMAKE_COMMAND} -E env PATH=${_NVCC_DIR}:$ENV{PATH}  ${CMAKE_COMMAND} -E environment # [=[echo $PATH]=]
    #BUILD_COMMAND ${CMAKE_COMMAND} -E env PATH=${_NVCC_DIR}:$ENV{PATH} ${CARGO} build --manifest-path=<SOURCE_DIR>/Cargo.toml --target-dir=<BINARY_DIR>
    BUILD_COMMAND ""
    INSTALL_COMMAND ${CMAKE_COMMAND} -E make_directory <INSTALL_DIR>/usr/local/include
        COMMAND ${CMAKE_COMMAND} -E create_symlink <SOURCE_DIR>/sql-ops/include <INSTALL_DIR>/usr/local/include/fast-interconnects
        COMMAND ${CMAKE_COMMAND} -E create_symlink <SOURCE_DIR>/sql-ops/cudautils <INSTALL_DIR>/usr/local/include/fast-interconnects_src
)

# Fetch paths
ExternalProject_Get_Property(fast_interconnects_src BINARY_DIR)
ExternalProject_Get_Property(fast_interconnects_src INSTALL_DIR)

# Collect include directories
set(FAST_INTERCONNECTS_INCLUDE_DIRS "")
list(APPEND FAST_INTERCONNECTS_INCLUDE_DIRS ${INSTALL_DIR}/usr/local/include)
# Files in fast-interconnects include local ones without any prefix, hence we need the following:
list(APPEND FAST_INTERCONNECTS_INCLUDE_DIRS ${INSTALL_DIR}/usr/local/include/fast-interconnects)

# Locate the constants.h header which is auto generated
#file(GLOB_RECURSE GENERATED_HEADERS CONFIGURE_DEPENDS "${BINARY_DIR}/debug/build/sql-ops-*/*.h")
#foreach(_HEADER_FILE ${GENERATED_HEADERS})
#    get_filename_component(_DIR ${_HEADER_FILE} PATH)
#    list(APPEND FAST_INTERCONNECTS_INCLUDE_DIRS ${_DIR})
#endforeach()
#list(REMOVE_DUPLICATES FAST_INTERCONNECTS_INCLUDE_DIRS)

message(STATUS "FAST_INTERCONNECTS_INCLUDE_DIRS: ${FAST_INTERCONNECTS_INCLUDE_DIRS}")

add_library(fast_interconnects INTERFACE)
set_property(TARGET fast_interconnects APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${FAST_INTERCONNECTS_INCLUDE_DIRS})

# Set up dependencies
add_dependencies(fast_interconnects fast_interconnects_src)
