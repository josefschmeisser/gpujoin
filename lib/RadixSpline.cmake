include(ExternalProject)
find_package(Git REQUIRED)

set(RADIXSPLINE_PREFIX RadixSpline)

# Get RadixSpline
ExternalProject_Add(
    RadixSpline
    PREFIX ${RADIXSPLINE_PREFIX}
    GIT_REPOSITORY "https://github.com/learnedsystems/RadixSpline.git"
    #GIT_TAG release-1.8.0
    TIMEOUT 10
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
)

ExternalProject_Get_Property(RadixSpline SOURCE_DIR)

set(RADIXSPLINE_INCLUDE_DIR ${SOURCE_DIR}/include)
