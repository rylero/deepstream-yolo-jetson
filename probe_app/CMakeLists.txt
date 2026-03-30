cmake_minimum_required(VERSION 3.16)
project(ds_nt_probe C)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GLIB      REQUIRED glib-2.0)

set(DS_ROOT /opt/nvidia/deepstream/deepstream)

include_directories(
    ${GSTREAMER_INCLUDE_DIRS}
    ${GLIB_INCLUDE_DIRS}
    ${DS_ROOT}/sources/includes          # gstnvdsmeta.h, nvdsmeta.h, etc.
    /usr/local/include                   # ntcore C headers
)

link_directories(
    ${DS_ROOT}/lib                       # nvdsgst_meta, nvds_meta, nvbufsurface
    /usr/local/lib                       # ntcore, wpiutil, wpinet
)

add_executable(ds_nt_probe ds_nt_probe.c)

target_compile_options(ds_nt_probe PRIVATE -Wall -Wextra)

target_link_libraries(ds_nt_probe
    ${GSTREAMER_LIBRARIES}
    ${GLIB_LIBRARIES}
    nvdsgst_meta
    nvds_meta
    nvbufsurface
    ntcore
    wpiutil
    wpinet
)

# Ensure shared libs in /usr/local/lib are found at runtime
set_target_properties(ds_nt_probe PROPERTIES
    INSTALL_RPATH "/usr/local/lib;${DS_ROOT}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)
