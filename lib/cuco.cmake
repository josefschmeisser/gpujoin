include(ExternalProject)
find_package(Git REQUIRED)

set(CUCO_PREFIX cuco)

# fetch cuco
ExternalProject_Add(
    cuco_src
    PREFIX ${CUCO_PREFIX}
    GIT_REPOSITORY "https://github.com/NVIDIA/cuCollections.git"
    #GIT_TAG "791a637d1787fb52f8855a52c400ce97cdca1ede"
    GIT_TAG "ee5c10456c7ad584c254152411ba3dc114537a6f"
    TIMEOUT 10
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
)

ExternalProject_Get_Property(cuco_src SOURCE_DIR)
set(CUCO_INCLUDE_DIR ${SOURCE_DIR}/include)
file(MAKE_DIRECTORY ${CUCO_INCLUDE_DIR}) # https://gitlab.kitware.com/cmake/cmake/-/issues/15052

# An interface library has no source files, however, we can use it
# to propagte include directories via INTERFACE_INCLUDE_DIRECTORIES to targets.
# By adding such a library through target_link_libraries() to a target,
# the library's interface include directories will be propagated to the target.
add_library(cuco INTERFACE)
set_target_properties(cuco PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${CUCO_INCLUDE_DIR})

# Dependencies
add_dependencies(cuco cuco_src)
