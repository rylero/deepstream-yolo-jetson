cmake_minimum_required(VERSION 3.16)
project(ds_nt_probe C CXX)  # CXX required by FetchContent internals

find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER     REQUIRED gstreamer-1.0)
pkg_check_modules(GSTREAMER_APP REQUIRED gstreamer-app-1.0)
pkg_check_modules(GLIB          REQUIRED glib-2.0)

set(DS_ROOT /opt/nvidia/deepstream/deepstream)

# ------------------------------------------------------------------ #
# WPILib via FetchContent (headers + arm64 shared libs)               #
# ------------------------------------------------------------------ #
include(FetchContent)

set(WPILIB_VER  "2025.3.2")
set(WPILIB_BASE "https://frcmaven.wpi.edu/artifactory/release/edu/wpi/first")

foreach(lib wpiutil wpinet ntcore)
    FetchContent_Declare(${lib}_headers
        URL "${WPILIB_BASE}/${lib}/${lib}-cpp/${WPILIB_VER}/${lib}-cpp-${WPILIB_VER}-headers.zip")
    FetchContent_Declare(${lib}_libs
        URL "${WPILIB_BASE}/${lib}/${lib}-cpp/${WPILIB_VER}/${lib}-cpp-${WPILIB_VER}-linuxarm64.zip")
endforeach()

FetchContent_MakeAvailable(
    wpiutil_headers wpiutil_libs
    wpinet_headers  wpinet_libs
    ntcore_headers  ntcore_libs)

# Locate each .so from its extracted zip directory
foreach(lib wpiutil wpinet ntcore)
    find_library(${lib}_SO NAMES ${lib}
        PATHS
            "${${lib}_libs_SOURCE_DIR}/linux/arm64/shared"
            "${${lib}_libs_SOURCE_DIR}/linuxarm64"
        NO_DEFAULT_PATH REQUIRED)
endforeach()

# ------------------------------------------------------------------ #
# ds_nt_probe executable                                              #
# ------------------------------------------------------------------ #
add_executable(ds_nt_probe ds_nt_probe.c)

target_compile_options(ds_nt_probe PRIVATE -Wall -Wextra)

target_include_directories(ds_nt_probe PRIVATE
    ${GSTREAMER_INCLUDE_DIRS}
    ${GSTREAMER_APP_INCLUDE_DIRS}
    ${GLIB_INCLUDE_DIRS}
    ${DS_ROOT}/sources/includes
    ${wpiutil_headers_SOURCE_DIR}
    ${wpinet_headers_SOURCE_DIR}
    ${ntcore_headers_SOURCE_DIR}
)

target_link_directories(ds_nt_probe PRIVATE
    ${DS_ROOT}/lib
)

target_link_libraries(ds_nt_probe PRIVATE
    ${GSTREAMER_LIBRARIES}
    ${GSTREAMER_APP_LIBRARIES}
    ${GLIB_LIBRARIES}
    nvdsgst_meta
    nvds_meta
    ${ntcore_SO}
    ${wpiutil_SO}
    ${wpinet_SO}
    pthread
)

# Copy WPILib .so files next to the binary so the RPATH finds them
foreach(lib wpiutil wpinet ntcore)
    add_custom_command(TARGET ds_nt_probe POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${${lib}_SO}" $<TARGET_FILE_DIR:ds_nt_probe>)
endforeach()

set_target_properties(ds_nt_probe PROPERTIES
    INSTALL_RPATH "$ORIGIN;${DS_ROOT}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)
