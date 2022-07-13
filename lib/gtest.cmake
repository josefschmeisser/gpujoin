include(ExternalProject)
find_package(Git REQUIRED)
find_package(Threads REQUIRED)

# Get googletest
ExternalProject_Add(
    googletest
    PREFIX "google/gtm"
    GIT_REPOSITORY "https://github.com/google/googletest.git"
    GIT_TAG release-1.10.0
    TIMEOUT 10
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
)

# Build gtest
ExternalProject_Add(
    gtest_src
    PREFIX "google/gtm"
    SOURCE_DIR "google/gtm/src/googletest/googletest"
    CMAKE_ARGS
        "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
        "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} -Wno-error=maybe-uninitialized"
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND ""
)

# Build gmock
ExternalProject_Add(
    gmock_src
    PREFIX "google/gtm"
    SOURCE_DIR "google/gtm/src/googletest/googlemock"
    CMAKE_ARGS
        "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
        "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} -Wno-error=maybe-uninitialized"
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND ""
)

# Prepare gtest
ExternalProject_Get_Property(gtest_src SOURCE_DIR)
ExternalProject_Get_Property(gtest_src BINARY_DIR)

# workaround for path inconsistencies between differen cmake versions
#if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18.0") # not sure when this breaking change was really introduced
if(${CMAKE_VERSION} VERSION_EQUAL "3.18.0") # not sure when this breaking change was really introduced
    set(library_path "${BINARY_DIR}")
else()
    set(library_path "${BINARY_DIR}/lib")
endif()

set(GTEST_INCLUDE_DIR ${SOURCE_DIR}/include )
set(GTEST_LIBRARY_PATH ${library_path}/libgtest.a)
file(MAKE_DIRECTORY ${GTEST_INCLUDE_DIR})

add_library(gtest STATIC IMPORTED)
set_property(TARGET gtest PROPERTY IMPORTED_LOCATION ${GTEST_LIBRARY_PATH})
set_property(TARGET gtest APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${GTEST_INCLUDE_DIR})

add_library(gtest_main STATIC IMPORTED)
set(GTEST_MAIN_LIBRARY_PATH ${library_path}/libgtest_main.a)
set_property(TARGET gtest_main PROPERTY IMPORTED_LOCATION ${GTEST_MAIN_LIBRARY_PATH})
set_property(TARGET gtest_main APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${GTEST_INCLUDE_DIR})

# Prepare gmock
ExternalProject_Get_Property(gtest_src SOURCE_DIR)
ExternalProject_Get_Property(gtest_src BINARY_DIR)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18.0") # not sure when this breaking change was really introduced
    set(library_path "${BINARY_DIR}")
else()
    #cmake_path(APPEND library_path "${BINARY_DIR}" "lib")
    set(library_path "${BINARY_DIR}/lib")
endif()

set(GMOCK_INCLUDE_DIR ${SOURCE_DIR}/include)
set(GMOCK_LIBRARY_PATH ${library_path}/libgmock.a)
file(MAKE_DIRECTORY ${GMOCK_INCLUDE_DIR})
add_library(gmock STATIC IMPORTED)
set_property(TARGET gmock PROPERTY IMPORTED_LOCATION ${GMOCK_LIBRARY_PATH})
set_property(TARGET gmock APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${GMOCK_INCLUDE_DIR})

# Dependencies
add_dependencies(gtest_src googletest)
add_dependencies(gmock_src googletest)
add_dependencies(gtest gtest_src)
add_dependencies(gtest_main gtest_src)
add_dependencies(gmock gmock_src)
