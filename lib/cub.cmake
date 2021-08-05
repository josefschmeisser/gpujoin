include(ExternalProject)
find_package(Git REQUIRED)

set(CUB_PREFIX cub)

# fetch cub
ExternalProject_Add(
    cub_src
    PREFIX ${CUB_PREFIX}
    GIT_REPOSITORY "https://github.com/NVlabs/cub.git"
    GIT_TAG "1.13.0"
    TIMEOUT 10
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
)

ExternalProject_Get_Property(cub_src SOURCE_DIR)
set(CUB_INCLUDE_DIR ${SOURCE_DIR})
file(MAKE_DIRECTORY ${CUB_INCLUDE_DIR}) # https://gitlab.kitware.com/cmake/cmake/-/issues/15052

# An interface library has no source files, however, we can use it
# to propagte includes dir via INTERFACE_INCLUDE_DIRECTORIES to targets.
# By adding such a library through target_link_libraries() to a target,
# the library's interface include directories will be propagated to target.
add_library(cub INTERFACE)
set_target_properties(cub PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${CUB_INCLUDE_DIR})

# Dependencies
add_dependencies(cub cub_src)
