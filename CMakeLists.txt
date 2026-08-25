cmake_minimum_required(VERSION 3.20)

project(learned_visual_pose LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)

if(APPLE AND NOT CMAKE_OSX_ARCHITECTURES)
    # CMake targets its OWN architecture on macOS, so an x86_64 cmake earlier on
    # PATH (a stale Intel Homebrew under /usr/local) silently produces x86_64
    # objects that an arm64 interpreter refuses to load. Follow the interpreter
    # rather than whichever cmake happened to win the PATH lookup.
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import platform; print(platform.machine())"
        OUTPUT_VARIABLE host_arch
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(CMAKE_OSX_ARCHITECTURES "${host_arch}" CACHE STRING "" FORCE)
endif()

# pybind11 keeps its CMake config inside site-packages, which is only on
# CMAKE_PREFIX_PATH while pip drives the build in its isolated environment. That
# environment is deleted afterwards, so `editable.rebuild` -- which re-runs CMake
# at import time -- cannot see it. Ask the interpreter first, and fall back to
# fetching pybind11 so a plain `pip install -e .` keeps working on rebuild.
set(PYBIND11_FINDPYTHON ON)

execute_process(
    COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_cmakedir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE pybind11_query
    ERROR_QUIET
)

if(pybind11_query EQUAL 0 AND EXISTS "${pybind11_cmakedir}")
    set(pybind11_DIR "${pybind11_cmakedir}" CACHE PATH "" FORCE)
    find_package(pybind11 CONFIG REQUIRED)
else()
    message(STATUS "pybind11 not importable from ${Python_EXECUTABLE}, fetching v3.1.0")
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v3.1.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# 5.0 is a hard floor, not a preference: the tests declare constexpr
# Eigen::Matrix3d, which is not a literal type before Eigen 5. Stating the
# version also steps past an older Eigen sitting earlier on the search path.
find_package(Eigen3 5.0 QUIET NO_MODULE)

if(NOT Eigen3_FOUND)
    message(STATUS "Eigen >= 5.0 not found, fetching 5.0.1")
    set(EIGEN_BUILD_TESTING OFF)
    set(EIGEN_BUILD_DOC OFF)
    set(EIGEN_BUILD_PKGCONFIG OFF)
    FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 5.0.1
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Eigen3)
endif()

add_library(
    geometry
    cpp/geometry/include/rotation_so3.hpp
    cpp/geometry/src/rotation_so3.cpp
    cpp/geometry/include/geometry.hpp
    cpp/geometry/include/pose_se3.hpp
    cpp/geometry/src/pose_se3.cpp
    cpp/geometry/include/utils.hpp
    cpp/geometry/include/pinhole_camera.hpp
    cpp/geometry/src/pinhole_camera.cpp
    cpp/geometry/include/pixel_to_pose.hpp
    cpp/geometry/src/pixel_to_pose.cpp
)

target_include_directories(
    geometry
    PUBLIC
      cpp/geometry/include
)

target_link_libraries(
    geometry
    PUBLIC
      Eigen3::Eigen
)

pybind11_add_module(_geometry cpp/bindings/geometry_py.cpp)
target_link_libraries(_geometry PRIVATE geometry)
install(TARGETS _geometry DESTINATION visual_pose)

# SKBUILD is set only when scikit-build-core is driving the build for a wheel.
# Everything below is dev-only: without the guard, every `pip install` would
# clone googletest and compile the scratch app.
if(NOT SKBUILD)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0
    )

    FetchContent_MakeAvailable(googletest)

    enable_testing()

    add_executable(
        learned_visual_pose
        cpp/app/main.cpp
    )

    target_link_libraries(
        learned_visual_pose
        PRIVATE
          geometry
    )

    add_executable(
        unit_tests
        cpp/tests/test_rotation_so3.cpp
        cpp/tests/test_pose_se3.cpp
        cpp/tests/test_pinhole_camera.cpp
        cpp/tests/test_pixel_to_pose.cpp
    )

    target_link_libraries(
        unit_tests
        PRIVATE
          GTest::gtest_main
          geometry
    )

    include(GoogleTest)
    gtest_discover_tests(unit_tests)

    # Regenerate the .pyi so the IDE can resolve the extension -- a compiled
    # module has no introspectable signatures, pybind11 keeps them in docstrings.
    # Runs against the freshly linked .so via PYTHONPATH, not the installed
    # editable copy, so the stub always matches what was just built.
    #
    # Dev builds only: a wheel build must not write into the source tree.
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import pybind11_stubgen"
        RESULT_VARIABLE stubgen_status
        OUTPUT_QUIET ERROR_QUIET
    )

    if(stubgen_status EQUAL 0)
        add_custom_command(
            TARGET _geometry POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E env
                    PYTHONPATH=$<TARGET_FILE_DIR:_geometry>
                    ${Python_EXECUTABLE} -m pybind11_stubgen _geometry
                    --output-dir ${CMAKE_CURRENT_SOURCE_DIR}/python/visual_pose
            COMMENT "Regenerating _geometry.pyi"
            VERBATIM
        )
    else()
        message(STATUS "pybind11-stubgen not found; _geometry.pyi will not be regenerated")
    endif()
endif()
