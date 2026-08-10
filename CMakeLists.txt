cmake_minimum_required(VERSION 3.20)

project(learned_visual_pose LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
)

FetchContent_MakeAvailable(googletest)

enable_testing()

find_package(Eigen3 REQUIRED)

add_library(
    geometry
    cpp/geometry/include/rotation_so3.hpp
    cpp/geometry/src/rotation_so3.cpp
    cpp/geometry/include/geometry.hpp
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
    cpp/tests/test_sanity.cpp
)

target_link_libraries(
    unit_tests
    PRIVATE
      GTest::gtest_main
      geometry
)

include(GoogleTest)
gtest_discover_tests(unit_tests)