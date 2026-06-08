cmake_minimum_required(VERSION 3.18)
project(gausseliminationcuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to Turing (Compute Capability 7.5 — GTX 16-series, RTX 20-series).
# Override at configure time:  cmake .. -DCMAKE_CUDA_ARCHITECTURES=86
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# GoogleTest via FetchContent — keeps the artifact dependency-free for users.
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(gauss_elim_bench src/gaussElimination.cu)
target_link_libraries(gauss_elim_bench PRIVATE GTest::gtest_main)
target_compile_options(gauss_elim_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

include(GoogleTest)
gtest_discover_tests(gauss_elim_bench
    PROPERTIES TIMEOUT 600
)
