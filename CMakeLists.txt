cmake_minimum_required(VERSION 3.15)
project(gausseliminationcuda LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to Turing (Compute Capability 7.5 - GTX 16-series, RTX 20-series).
# CMake 3.18+ has first-class CUDA architecture support. Visual Studio 2019
# can use a bundled CMake 3.15, so fall back to an explicit NVCC flag there.
set(GAUSS_CUDA_ARCHITECTURE "75" CACHE STRING "CUDA architecture, e.g. 75 for GTX 1650 or 86 for RTX 30-series")
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES ${GAUSS_CUDA_ARCHITECTURE})
    endif()
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# GoogleTest via FetchContent - keeps the artifact dependency-free for users.
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
target_link_libraries(gauss_elim_bench PRIVATE GTest::gtest CUDA::cusolver CUDA::cublas)
target_include_directories(gauss_elim_bench PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

set(GAUSS_CUDA_COMPILE_OPTIONS --expt-relaxed-constexpr)
if(CMAKE_VERSION VERSION_LESS 3.18)
    list(APPEND GAUSS_CUDA_COMPILE_OPTIONS
        "--generate-code=arch=compute_${GAUSS_CUDA_ARCHITECTURE},code=sm_${GAUSS_CUDA_ARCHITECTURE}"
    )
endif()
target_compile_options(gauss_elim_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${GAUSS_CUDA_COMPILE_OPTIONS}>
)

include(GoogleTest)
gtest_discover_tests(gauss_elim_bench
    PROPERTIES TIMEOUT 600
)
