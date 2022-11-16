include(ExternalProject)
find_package(Git REQUIRED)
#find_package(Rust REQUIRED) # TODO

find_program(CARGO "cargo")
IF(NOT CARGO)
    message(FATAL_ERROR "cargo not found!")
ENDIF()

set(FAST_INTERCONNECTS_PREFIX "fast-interconnects")

# Get FAST_INTERCONNECTS
ExternalProject_Add(
    fast_interconnects_src
    PREFIX ${FAST_INTERCONNECTS_PREFIX}
    GIT_REPOSITORY "git@github.com:TU-Berlin-DIMA/fast-interconnects.git"
    GIT_TAG "84c181a"
    TIMEOUT 10
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${CARGO} build --manifest-path=<SOURCE_DIR>/Cargo.toml --target-dir=<BINARY_DIR>
    INSTALL_COMMAND ""
)

# Fetch paths
ExternalProject_Get_Property(fast_interconnects_src SOURCE_DIR)
ExternalProject_Get_Property(fast_interconnects_src BINARY_DIR)

#message(STATUS "FAST_INTERCONNECTS INSTALL_DIR: ${INSTALL_DIR} BINARY_DIR: ${BINARY_DIR} SOURCE_DIR: ${SOURCE_DIR}")

# Locate the constants.h header which is auto generated
file(GLOB_RECURSE GENERATED_HEADERS CONFIGURE_DEPENDS "${BINARY_DIR}/debug/build/sql-ops-*/*.h")

set(FAST_INTERCONNECTS_INCLUDE_DIRS "")
list(APPEND FAST_INTERCONNECTS_INCLUDE_DIRS ${SOURCE_DIR})
foreach(_HEADER_FILE ${GENERATED_HEADERS})
    get_filename_component(_DIR ${_HEADER_FILE} PATH)
    #message(STATUS "header: ${_HEADER_FILE} dir: ${_DIR}")
    list(APPEND FAST_INTERCONNECTS_INCLUDE_DIRS ${_DIR})
endforeach()
#list (REMOVE_DUPLICATES Foo_INCLUDE_DIRS)


message(STATUS "FAST_INTERCONNECTS_INCLUDE_DIRS: ${FAST_INTERCONNECTS_INCLUDE_DIRS}")
#set(FAST_INTERCONNECTS_INCLUDE_DIR ${INSTALL_DIR}/usr/local/include)
#file(MAKE_DIRECTORY ${FAST_INTERCONNECTS_INCLUDE_DIR})

add_library(fast_interconnects INTERFACE)
#set_property(TARGET fast_interconnects PROPERTY IMPORTED_LOCATION ${FAST_INTERCONNECTS_LIBRARY_PATH})
set_property(TARGET fast_interconnects APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${FAST_INTERCONNECTS_INCLUDE_DIRS})

# Dependencies
add_dependencies(fast_interconnects fast_interconnects_src)
