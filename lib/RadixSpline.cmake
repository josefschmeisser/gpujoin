include(ExternalProject)
find_package(Git REQUIRED)

set(RADIXSPLINE_PREFIX RadixSpline)

# Get RadixSpline
ExternalProject_Add(
    radixspline_src
    PREFIX ${RADIXSPLINE_PREFIX}
    GIT_REPOSITORY "https://github.com/learnedsystems/RadixSpline.git"
    TIMEOUT 10
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
)

ExternalProject_Get_Property(radixspline_src SOURCE_DIR)
set(RADIXSPLINE_INCLUDE_DIR ${SOURCE_DIR}/include)
file(MAKE_DIRECTORY ${RADIXSPLINE_INCLUDE_DIR}) # https://gitlab.kitware.com/cmake/cmake/-/issues/15052

# An interface library has no source files, however, we can use it
# to propagte include directories via INTERFACE_INCLUDE_DIRECTORIES to targets.
# By adding such a library through target_link_libraries() to a target,
# the library's interface include directories will be propagated to the target.
add_library(radixspline INTERFACE)
set_target_properties(radixspline PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${RADIXSPLINE_INCLUDE_DIR})

# Dependencies
add_dependencies(radixspline radixspline_src)
