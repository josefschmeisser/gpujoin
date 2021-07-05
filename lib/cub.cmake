include(ExternalProject)
find_package(Git REQUIRED)

set(CUB_PREFIX cub)

# fetch cub
ExternalProject_Add(
    cub
    PREFIX ${CUB_PREFIX}
    GIT_REPOSITORY "https://github.com/NVlabs/cub.git"
    GIT_TAG "1.13.0"
    TIMEOUT 10
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
)

ExternalProject_Get_Property(cub SOURCE_DIR)

set(CUB_INCLUDE_DIR ${SOURCE_DIR})
