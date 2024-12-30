include(ExternalProject)
find_package(Git REQUIRED)
find_package(Threads REQUIRED)

set(ONETBB_PREFIX "intel/onetbb")

# Get oneTBB
ExternalProject_Add(
    onetbb_src
    PREFIX ${ONETBB_PREFIX}
    GIT_REPOSITORY "https://github.com/oneapi-src/oneTBB.git"
    GIT_TAG v2021.9.0
    TIMEOUT 10
    CMAKE_ARGS
        "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
        "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DTBB_TEST=OFF"
    #BUILD_COMMAND   ${CMAKE_COMMAND} -E env CMAKE_BUILD_PARALLEL_LEVEL=`$(nproc)` ${CMAKE_COMMAND} --build .
    BUILD_COMMAND   ${CMAKE_COMMAND} -E env CMAKE_BUILD_PARALLEL_LEVEL=${NCORES} ${CMAKE_COMMAND} --build .
    INSTALL_COMMAND ${CMAKE_COMMAND} -E env DESTDIR=${CMAKE_BINARY_DIR}/${ONETBB_PREFIX} ${CMAKE_MAKE_PROGRAM} install
    UPDATE_COMMAND ""
)

# Prepare oneTBB
ExternalProject_Get_Property(onetbb_src INSTALL_DIR)

message(STATUS "oneTBB INSTALL_DIR: ${INSTALL_DIR}")

set(ONETBB_INCLUDE_DIR ${INSTALL_DIR}/usr/local/include)
set(ONETBB_LIBRARY_PATH ${INSTALL_DIR}/usr/local/lib/libtbb.so)
file(MAKE_DIRECTORY ${ONETBB_INCLUDE_DIR})

add_library(onetbb SHARED IMPORTED)
set_property(TARGET onetbb PROPERTY IMPORTED_LOCATION ${ONETBB_LIBRARY_PATH})
set_property(TARGET onetbb APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${ONETBB_INCLUDE_DIR})

# Dependencies
add_dependencies(onetbb onetbb_src)
