cmake_minimum_required(VERSION 3.24)
project(libnrn VERSION 0.1.0 LANGUAGES CXX CUDA)

# ============================================================================
# Build options
# ============================================================================
option(NRN_BUILD_TESTS    "Build tests"    ON)
option(NRN_BUILD_EXAMPLES "Build examples" ON)

set(NRN_CUDA_ARCHITECTURES "80;86;89;90" CACHE STRING "CUDA architectures to target")

# ============================================================================
# C++ and CUDA standards
# ============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Dependencies
# ============================================================================
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(yaml-cpp REQUIRED)

if(NRN_BUILD_TESTS)
    find_package(GTest REQUIRED)
    enable_testing()
endif()

# ============================================================================
# Subdirectories
# ============================================================================
add_subdirectory(src)

if(NRN_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(NRN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
