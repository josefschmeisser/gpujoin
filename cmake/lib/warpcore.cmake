include(ExternalProject)
find_package(Git REQUIRED)

set(WARPCORE_PREFIX warpcore)

# fetch warpcore
ExternalProject_Add(
    warpcore_src
    PREFIX ${WARPCORE_PREFIX}
    GIT_REPOSITORY "https://github.com/sleeepyjack/warpcore.git"
    #GIT_TAG "ee5c10456c7ad584c254152411ba3dc114537a6f"
    TIMEOUT 10
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
)

ExternalProject_Get_Property(warpcore_src SOURCE_DIR)
set(WARPCORE_INCLUDE_DIR ${SOURCE_DIR}/include)
file(MAKE_DIRECTORY ${WARPCORE_INCLUDE_DIR}) # https://gitlab.kitware.com/cmake/cmake/-/issues/15052

# An interface library has no source files, however, we can use it
# to propagte include directories via INTERFACE_INCLUDE_DIRECTORIES to targets.
# By adding such a library through target_link_libraries() to a target,
# the library's interface include directories will be propagated to the target.
add_library(warpcore INTERFACE)
set_target_properties(warpcore PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${WARPCORE_INCLUDE_DIR})

# Dependencies
add_dependencies(warpcore warpcore_src)
