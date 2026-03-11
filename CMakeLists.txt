cmake_minimum_required(VERSION 3.18)
project(reaper_stem_separation_plugin VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -flto -DNDEBUG")
endif()

# -- Vendor paths --
set(DEMUCS_ONNX_DIR "${CMAKE_SOURCE_DIR}/vendor/demucs.onnx")
set(WDL_DIR          "${CMAKE_SOURCE_DIR}/vendor/wdl")
set(SDK_DIR          "${CMAKE_SOURCE_DIR}/vendor/reaper-sdk/sdk")

if(NOT WIN32)
    # reaper_plugin.h includes "../WDL/swell/swell.h" relative to SDK_DIR
    if(NOT EXISTS "${SDK_DIR}/../WDL")
        file(CREATE_LINK "../wdl/WDL" "${SDK_DIR}/../WDL" SYMBOLIC)
    endif()
endif()

# -- ONNX Runtime --
# Prefer ORT_PREFIX for CI reproducibility; fall back to system search
if(ORT_PREFIX)
    set(ORT_LIB_DIR "${ORT_PREFIX}/lib")
    set(ORT_INC_DIR "${ORT_PREFIX}/include")
    find_library(ORT_LIB NAMES onnxruntime PATHS "${ORT_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
    set(ORT_INCLUDE "${ORT_INC_DIR}")

    # ORT release tarballs have headers flat in include/.
    # Code uses #include <onnxruntime/core/session/onnxruntime_cxx_api.h>,
    # so create the nested symlink structure if it doesn't exist.
    set(_ORT_SESSION_DIR "${ORT_INC_DIR}/onnxruntime/core/session")
    if(NOT EXISTS "${_ORT_SESSION_DIR}/onnxruntime_cxx_api.h")
        file(MAKE_DIRECTORY "${_ORT_SESSION_DIR}")
        file(GLOB _ORT_HDRS "${ORT_INC_DIR}/onnxruntime_*.h"
                             "${ORT_INC_DIR}/cpu_provider_factory.h"
                             "${ORT_INC_DIR}/provider_options.h")
        foreach(_h ${_ORT_HDRS})
            get_filename_component(_hname "${_h}" NAME)
            file(CREATE_LINK "${_h}" "${_ORT_SESSION_DIR}/${_hname}" SYMBOLIC)
        endforeach()
    endif()
else()
    find_library(ORT_LIB NAMES onnxruntime REQUIRED)
    find_path(ORT_INCLUDE NAMES onnxruntime_cxx_api.h REQUIRED)
endif()

# -- Eigen3 --
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    set(EIGEN3_INCLUDE_DIR "${DEMUCS_ONNX_DIR}/vendor/eigen" CACHE PATH "Eigen3 include dir")
endif()

# -- libnyquist --
set(LIBNYQUIST_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
add_subdirectory(${DEMUCS_ONNX_DIR}/vendor/libnyquist ${CMAKE_CURRENT_BINARY_DIR}/_libnyquist)

# -- Gather sources --
file(GLOB DEMUCS_ONNX_SOURCES "${DEMUCS_ONNX_DIR}/src/*.cpp")

set(PLUGIN_SOURCES
    src/main.cpp
    src/actions.cpp
    src/dialog.cpp
    src/model_manager.cpp
    src/separator.cpp
    src/audio_io.cpp
    src/tracks.cpp
)

if(APPLE)
    list(APPEND PLUGIN_SOURCES
        src/swell_resgen.cpp
        ${WDL_DIR}/WDL/swell/swell-modstub.mm
    )
elseif(UNIX)
    list(APPEND PLUGIN_SOURCES
        src/swell_resgen.cpp
        ${WDL_DIR}/WDL/swell/swell-modstub-generic.cpp
    )
endif()

# -- Shared library target --
add_library(reaper_stem_separation_plugin SHARED
    ${PLUGIN_SOURCES}
    ${DEMUCS_ONNX_SOURCES}
)

set_target_properties(reaper_stem_separation_plugin PROPERTIES
    PREFIX ""
    OUTPUT_NAME "reaper_stem_separation_plugin"
)

if(DOWNLOAD_REPO)
    target_compile_definitions(reaper_stem_separation_plugin PRIVATE
        DOWNLOAD_REPO="${DOWNLOAD_REPO}"
    )
endif()

if(DOWNLOAD_TAG)
    target_compile_definitions(reaper_stem_separation_plugin PRIVATE
        DOWNLOAD_TAG="${DOWNLOAD_TAG}"
    )
endif()

if(NOT WIN32)
    target_compile_definitions(reaper_stem_separation_plugin PRIVATE
        SWELL_PROVIDED_BY_APP
        NOMINMAX
    )
endif()

if(WIN32)
    target_compile_definitions(reaper_stem_separation_plugin PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
    target_sources(reaper_stem_separation_plugin PRIVATE src/dialog_win32.rc)
endif()

target_include_directories(reaper_stem_separation_plugin PRIVATE
    ${SDK_DIR}
    ${WDL_DIR}
    ${DEMUCS_ONNX_DIR}/src
    ${DEMUCS_ONNX_DIR}/vendor/libnyquist/include
    ${ORT_INCLUDE}
    ${CMAKE_SOURCE_DIR}/src
)


if(Eigen3_FOUND)
    target_link_libraries(reaper_stem_separation_plugin PRIVATE Eigen3::Eigen)
else()
    target_include_directories(reaper_stem_separation_plugin PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()

target_link_libraries(reaper_stem_separation_plugin PRIVATE
    ${ORT_LIB}
    libnyquist
)

if(UNIX AND NOT APPLE)
    target_link_libraries(reaper_stem_separation_plugin PRIVATE pthread dl)
elseif(APPLE)
    find_library(COCOA_FW Cocoa)
    find_library(CARBON_FW Carbon)
    target_link_libraries(reaper_stem_separation_plugin PRIVATE pthread dl ${COCOA_FW} ${CARBON_FW})
endif()

# RPATH: look for ORT libs in reaper-stem-separation-plugin/ next to the plugin
if(APPLE)
    set_target_properties(reaper_stem_separation_plugin PROPERTIES
        SUFFIX ".dylib"
        INSTALL_RPATH "@loader_path/reaper-stem-separation-plugin"
        BUILD_RPATH   "@loader_path/reaper-stem-separation-plugin"
    )
elseif(UNIX)
    set_target_properties(reaper_stem_separation_plugin PROPERTIES
        INSTALL_RPATH "$ORIGIN/reaper-stem-separation-plugin"
        BUILD_RPATH   "$ORIGIN/reaper-stem-separation-plugin"
    )
endif()

install(TARGETS reaper_stem_separation_plugin LIBRARY DESTINATION . RUNTIME DESTINATION .)
