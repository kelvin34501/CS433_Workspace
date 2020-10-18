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

# for tutorial
add_executable(cuda_test src/cuda_test.cu)
target_compile_features(cuda_test PUBLIC cxx_std_11)

add_executable(__learn src/__learn.cu)
target_compile_features(__learn PRIVATE cxx_std_11)

# compile things into lib
add_library(conv_cpu include/conv_cpu.h src/conv_cpu.cpp)
target_compile_features(conv_cpu PUBLIC cxx_std_17)
target_include_directories(conv_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(conv_gpu include/conv_gpu.h src/conv_gpu.cu)
# i don't think nvcc could use c++17... fallback to c++11
target_compile_features(conv_gpu PUBLIC cxx_std_11) 
target_include_directories(conv_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# testing things
add_executable(hw1p1_test test/hw1p1.cpp)
target_compile_features(hw1p1_test PUBLIC cxx_std_17)
target_compile_options(hw1p1_test PUBLIC
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -pedantic>
)
target_include_directories(hw1p1_test PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
)
target_link_libraries(hw1p1_test PUBLIC 
    gtest_main
    conv_cpu
    conv_gpu
)

add_executable(hw1p2_test test/hw1p2.cpp)
target_compile_features(hw1p2_test PUBLIC cxx_std_17)
target_compile_options(hw1p2_test PUBLIC
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -pedantic>
)
target_include_directories(hw1p2_test PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
)
target_link_libraries(hw1p2_test PUBLIC 
    gtest_main
    conv_cpu
    conv_gpu
)

# benchmarking things
add_executable(hw1p1_benchmark benchmark/hw1p1.cpp)
target_compile_features(hw1p1_benchmark PUBLIC cxx_std_17)
target_compile_options(hw1p1_benchmark PUBLIC
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -pedantic>
)
target_include_directories(hw1p1_benchmark PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
)
target_link_libraries(hw1p1_benchmark PUBLIC 
    benchmark::benchmark
    conv_cpu
    conv_gpu
)
