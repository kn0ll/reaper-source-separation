cmake_minimum_required(VERSION 3.18)
project(reaper_stem_separation_plugin VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    add_compile_options(/W4 /WX /Zc:preprocessor)
else()
    # -ffast-math: intentional for audio DSP performance (relaxes IEEE 754 strictness)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -flto -DNDEBUG")
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
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
if(MSVC)
    target_compile_options(libnyquist PRIVATE /w)
else()
    target_compile_options(libnyquist PRIVATE -w)
endif()

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
    src/ort_provider.cpp
    src/demucs_backend.cpp
    src/roformer_backend.cpp
    src/stft.cpp
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

# -- Vendor demucs.onnx (upstream submodule, compiled without project warnings) --
add_library(demucs_onnx_vendor OBJECT ${DEMUCS_ONNX_SOURCES})
set_target_properties(demucs_onnx_vendor PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(demucs_onnx_vendor PRIVATE
    ${DEMUCS_ONNX_DIR}/src
    ${ORT_INCLUDE}
)
if(Eigen3_FOUND)
    target_link_libraries(demucs_onnx_vendor PRIVATE Eigen3::Eigen)
else()
    target_include_directories(demucs_onnx_vendor PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()
if(MSVC)
    target_compile_options(demucs_onnx_vendor PRIVATE /w)
else()
    target_compile_options(demucs_onnx_vendor PRIVATE -w)
endif()

# -- Shared library target --
add_library(reaper_stem_separation_plugin SHARED
    ${PLUGIN_SOURCES}
    $<TARGET_OBJECTS:demucs_onnx_vendor>
)

set_target_properties(reaper_stem_separation_plugin PROPERTIES
    PREFIX ""
    OUTPUT_NAME "reaper_stem_separation_plugin"
)

# Suppress warnings on vendor source files / vendor-macro-generated files
if(APPLE)
    set_source_files_properties(${WDL_DIR}/WDL/swell/swell-modstub.mm PROPERTIES COMPILE_OPTIONS "-w")
    set_source_files_properties(src/swell_resgen.cpp PROPERTIES COMPILE_OPTIONS "-w")
elseif(UNIX)
    set_source_files_properties(${WDL_DIR}/WDL/swell/swell-modstub-generic.cpp PROPERTIES COMPILE_OPTIONS "-w")
    set_source_files_properties(src/swell_resgen.cpp PROPERTIES COMPILE_OPTIONS "-w")
endif()

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
    ${CMAKE_SOURCE_DIR}/src
)
target_include_directories(reaper_stem_separation_plugin SYSTEM PRIVATE
    ${SDK_DIR}
    ${WDL_DIR}
    ${DEMUCS_ONNX_DIR}/src
    ${DEMUCS_ONNX_DIR}/vendor/libnyquist/include
    ${ORT_INCLUDE}
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
