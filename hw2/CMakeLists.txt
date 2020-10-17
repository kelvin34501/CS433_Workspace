cmake_minimum_required(VERSION 3.10.0)
project(hw2 VERSION 0.1.0 LANGUAGES CXX)

enable_language(CUDA)

# set build type to Release
set(CMAKE_BUILD_TYPE Release)
message(STATUS "Cmake Build Type: ${CMAKE_BUILD_TYPE}")
# this is a fix for msvc
if(MSVC)
    message(STATUS "Apply fix on MSVC (workaround)")
    set(CompilerFlags
            CMAKE_CXX_FLAGS
            CMAKE_CXX_FLAGS_DEBUG
            CMAKE_CXX_FLAGS_RELEASE
            CMAKE_CXX_FLAGS_RELWITHDEBINFO
            CMAKE_CXX_FLAGS_MINSIZEREL
            CMAKE_C_FLAGS
            CMAKE_C_FLAGS_DEBUG
            CMAKE_C_FLAGS_RELEASE
            CMAKE_C_FLAGS_RELWITHDEBINFO
            CMAKE_C_FLAGS_MINSIZEREL
            )
    foreach(CompilerFlag ${CompilerFlags})
        string(REPLACE "/MD" "/MT" ${CompilerFlag} "${${CompilerFlag}}")
        message(STATUS "\tReplaced Flags: ${CompilerFlag}: ${${CompilerFlag}}")
    endforeach()
endif()

# set a few vars
set(WS_DIR_REL ${CMAKE_CURRENT_SOURCE_DIR}/..)
get_filename_component(WS_DIR ${WS_DIR_REL} ABSOLUTE)
message(STATUS "Workspace Directory: ${WS_DIR}")

# find google test [bundled]
set(HW1_GOOGLETEST_DIR ${WS_DIR}/external/googletest)
message(STATUS "googletest Directory: ${HW1_BENCHMARK_DIR}")
add_subdirectory(${HW1_GOOGLETEST_DIR} ${CMAKE_CURRENT_BINARY_DIR}/external/googletest)

# find google benchmark [bundled]
set(HW1_BENCHMARK_DIR ${WS_DIR}/external/benchmark)
message(STATUS "benchmark Directory: ${HW1_BENCHMARK_DIR}")
add_subdirectory(${HW1_BENCHMARK_DIR} ${CMAKE_CURRENT_BINARY_DIR}/external/benchmark)

# test things up
add_executable(cuda_test cuda_test.cu)
target_compile_features(cuda_test PUBLIC cxx_std_11)