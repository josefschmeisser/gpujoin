include(ExternalProject)
find_package(Git REQUIRED)

set(CUCOLLECTIONS_PREFIX cuCollections)

# Get RadixSpline
ExternalProject_Add(
    cucollections_src
    PREFIX ${CUCOLLECTIONS_PREFIX}
    GIT_REPOSITORY "https://github.com/NVIDIA/cuCollections.git"
    GIT_TAG "dev"
    TIMEOUT 10
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
)

ExternalProject_Get_Property(cucollections_src SOURCE_DIR)
set(CUCOLLECTIONS_INCLUDE_DIR ${SOURCE_DIR}/include)
file(MAKE_DIRECTORY ${CUCOLLECTIONS_INCLUDE_DIR}) # https://gitlab.kitware.com/cmake/cmake/-/issues/15052

# An interface library has no source files, however, we can use it
# to propagte include directories via INTERFACE_INCLUDE_DIRECTORIES to targets.
# By adding such a library through target_link_libraries() to a target,
# the library's interface include directories will be propagated to the target.
add_library(cuCollections INTERFACE)
set_target_properties(cuCollections PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${CUCOLLECTIONS_INCLUDE_DIR})

# Dependencies
add_dependencies(cuCollections cucollections_src)
