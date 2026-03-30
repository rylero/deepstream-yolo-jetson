cmake_minimum_required(VERSION 3.16)
project(ds_nt_probe C CXX)  # CXX required by FetchContent/WPILib

set(CMAKE_CXX_STANDARD 17)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GLIB      REQUIRED glib-2.0)

set(DS_ROOT /opt/nvidia/deepstream/deepstream)

# ------------------------------------------------------------------ #
# WPILib via FetchContent                                             #
# ------------------------------------------------------------------ #
include(FetchContent)

set(WPILIB_VER   "2025.3.2")
set(WPILIB_BASE  "https://frcmaven.wpi.edu/artifactory/release/edu/wpi/first")

function(fetch_wpilib_lib NAME)
    FetchContent_Declare(${NAME}_headers
        URL "${WPILIB_BASE}/${NAME}/${NAME}-cpp/${WPILIB_VER}/${NAME}-cpp-${WPILIB_VER}-headers.zip")
    FetchContent_MakeAvailable(${NAME}_headers)

    FetchContent_Declare(${NAME}_libs
        URL "${WPILIB_BASE}/${NAME}/${NAME}-cpp/${WPILIB_VER}/${NAME}-cpp-${WPILIB_VER}-linuxarm64.zip")
    FetchContent_MakeAvailable(${NAME}_libs)

    add_library(${NAME}_inc INTERFACE)
    target_include_directories(${NAME}_inc INTERFACE "${${NAME}_headers_SOURCE_DIR}")

    add_library(${NAME} SHARED IMPORTED)
    find_library(${NAME}_LIB_PATH NAMES ${NAME}
        PATHS
            "${${NAME}_libs_SOURCE_DIR}/linux/arm64/shared"
            "${${NAME}_libs_SOURCE_DIR}/linuxarm64"
        NO_DEFAULT_PATH REQUIRED)
    set_target_properties(${NAME} PROPERTIES IMPORTED_LOCATION "${${NAME}_LIB_PATH}")
    target_link_libraries(${NAME} INTERFACE ${NAME}_inc)
endfunction()

fetch_wpilib_lib(wpiutil)
fetch_wpilib_lib(wpinet)
fetch_wpilib_lib(ntcore)
target_link_libraries(wpinet INTERFACE wpiutil)
target_link_libraries(ntcore INTERFACE wpinet wpiutil)

# ------------------------------------------------------------------ #
# ds_nt_probe executable                                              #
# ------------------------------------------------------------------ #
add_executable(ds_nt_probe ds_nt_probe.c)

target_compile_options(ds_nt_probe PRIVATE -Wall -Wextra)

target_include_directories(ds_nt_probe PRIVATE
    ${GSTREAMER_INCLUDE_DIRS}
    ${GLIB_INCLUDE_DIRS}
    ${DS_ROOT}/sources/includes          # gstnvdsmeta.h, nvdsmeta.h
)

target_link_directories(ds_nt_probe PRIVATE
    ${DS_ROOT}/lib                       # nvdsgst_meta, nvds_meta, nvbufsurface
)

target_link_libraries(ds_nt_probe PRIVATE
    ${GSTREAMER_LIBRARIES}
    ${GLIB_LIBRARIES}
    nvdsgst_meta
    nvds_meta
    nvbufsurface
    ntcore
    wpiutil
    wpinet
)

# Copy WPILib .so files next to the binary so they're found at runtime
foreach(libname wpiutil wpinet ntcore)
    get_target_property(lib_loc ${libname} IMPORTED_LOCATION)
    add_custom_command(TARGET ds_nt_probe POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${lib_loc}" $<TARGET_FILE_DIR:ds_nt_probe>)
endforeach()

set_target_properties(ds_nt_probe PROPERTIES
    INSTALL_RPATH "$ORIGIN;${DS_ROOT}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)
