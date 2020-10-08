cmake_minimum_required(VERSION 3.10.0)
project(hw1 VERSION 0.1.0)

# set build type to RelWithDebInfo
set(CMAKE_BUILD_TYPE RelWithDebInfo)
message(STATUS "Cmake Build Type: ${CMAKE_BUILD_TYPE}")

# set a few vars
set(WS_DIR_REL ${CMAKE_CURRENT_SOURCE_DIR}/..)
get_filename_component(WS_DIR ${WS_DIR_REL} ABSOLUTE)
message(STATUS "Workspace Directory: ${WS_DIR}")

# find openmp
find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "OpenMP Not Found.")
endif()

# find google test [bundled]
set(HW1_GOOGLETEST_DIR ${WS_DIR}/external/googletest)
message(STATUS "googletest Directory: ${HW1_BENCHMARK_DIR}")
add_subdirectory(${HW1_GOOGLETEST_DIR} ${CMAKE_CURRENT_BINARY_DIR}/external/googletest)

# find google benchmark [bundled]
set(HW1_BENCHMARK_DIR ${WS_DIR}/external/benchmark)
message(STATUS "benchmark Directory: ${HW1_BENCHMARK_DIR}")
add_subdirectory(${HW1_BENCHMARK_DIR} ${CMAKE_CURRENT_BINARY_DIR}/external/benchmark)

# benchmarking
add_executable(hw1p1_benchmark benchmark/hw1p1_benchmark.cpp)
target_compile_features(hw1p1_benchmark PUBLIC cxx_std_17)
target_compile_options(hw1p1_benchmark PUBLIC
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -pedantic>
)
target_include_directories(hw1p1_benchmark PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
    ${benchmark_INCLUDE_DIRS}
)
target_link_libraries(hw1p1_benchmark PUBLIC 
    OpenMP::OpenMP_CXX
    benchmark::benchmark
)
