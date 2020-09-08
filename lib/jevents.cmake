include(ExternalProject)
find_package(Git REQUIRED)

set(JEVENTS_PREFIX "pmu-tools/jevents")

if (CMAKE_VERSION VERSION_LESS 3.15.0)
    # Get jevents
    ExternalProject_Add(
        jevents_src
        PREFIX "pmu-tools/jevents"
        GIT_REPOSITORY "https://github.com/andikleen/pmu-tools.git"
        GIT_TAG r109
        TIMEOUT 10
#        SOURCE_SUBDIR "jevents"
        BUILD_IN_SOURCE True
        CONFIGURE_COMMAND echo "No configuration necessary."
        BUILD_COMMAND     cd jevents && ${CMAKE_MAKE_PROGRAM} all
        INSTALL_COMMAND   cd jevents && ${CMAKE_COMMAND} -E env DESTDIR=${CMAKE_BINARY_DIR}/${JEVENTS_PREFIX} ${CMAKE_MAKE_PROGRAM} install
        UPDATE_COMMAND ""
    )
else()
    # Get jevents
    ExternalProject_Add(
        jevents_src
        PREFIX "pmu-tools/jevents"
        GIT_REPOSITORY "https://github.com/andikleen/pmu-tools.git"
        GIT_TAG r109
        TIMEOUT 10
        SOURCE_SUBDIR "jevents"
        BUILD_IN_SOURCE True
        CONFIGURE_COMMAND echo "No configuration necessary."
        BUILD_COMMAND     ${CMAKE_MAKE_PROGRAM} all
        INSTALL_COMMAND   ${CMAKE_COMMAND} -E env DESTDIR=${CMAKE_BINARY_DIR}/${JEVENTS_PREFIX} ${CMAKE_MAKE_PROGRAM} install
        UPDATE_COMMAND ""
    )
endif()

# Prepare jevents
ExternalProject_Get_Property(jevents_src install_dir)
set(JEVENTS_INCLUDE_DIR ${install_dir}/usr/local/include)
set(JEVENTS_LIBRARY_PATH ${install_dir}/usr/local/lib64/libjevents.a) # NOTE: "lib64" is hardcoded in jevents's Makefile
file(MAKE_DIRECTORY ${JEVENTS_INCLUDE_DIR})
add_library(jevents STATIC IMPORTED)
set_property(TARGET jevents PROPERTY IMPORTED_LOCATION ${JEVENTS_LIBRARY_PATH})
set_property(TARGET jevents APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${JEVENTS_INCLUDE_DIR})

# Dependencies
add_dependencies(jevents jevents_src)
