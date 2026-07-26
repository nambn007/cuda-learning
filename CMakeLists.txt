# ============================================================
# CUDA Learning - root build file
# ============================================================
#   cmake -S . -B build
#   cmake --build build -j$(nproc)
#   ctest --test-dir build --output-on-failure
#
# Useful options:
#   -DCUDA_ARCH=native      auto-detect this machine's GPU (default)
#   -DCUDA_ARCH=portable    fat binary for sm_70/75/80/86/89
#   -DCUDA_ARCH=86          one specific architecture
#   -DCMAKE_BUILD_TYPE=Debug  adds -G for cuda-gdb (kernels run ~10x slower)
#   -DCL_ENABLE_OPTIONAL=ON   build exercises with external deps
# ============================================================

cmake_minimum_required(VERSION 3.20)

# ------------------------------------------------------------
# GPU architecture selection
# ------------------------------------------------------------
# This must happen BEFORE project(), because enabling the CUDA
# language bakes in the default architecture list.
#
# Why this matters: nvcc compiles to SASS for the architectures you
# name. Running a binary on a newer GPU works only through PTX JIT
# (slower first launch, no architecture-specific optimisations);
# running on an older GPU does not work at all.
set(CUDA_ARCH "native" CACHE STRING
    "GPU architecture: native | portable | explicit list such as 86 or 75;86")

if(CUDA_ARCH STREQUAL "native")
    # Ask the driver directly - more reliable than CMake's probe and
    # it lets us print a helpful message.
    find_program(CL_NVIDIA_SMI nvidia-smi)
    set(_detected "")
    if(CL_NVIDIA_SMI)
        execute_process(
            COMMAND ${CL_NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE _smi_out
            ERROR_VARIABLE _smi_err
            RESULT_VARIABLE _smi_rc
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(_smi_rc EQUAL 0 AND _smi_out)
            string(REPLACE "\n" ";" _caps "${_smi_out}")
            foreach(_cap IN LISTS _caps)
                string(STRIP "${_cap}" _cap)
                string(REPLACE "." "" _cap "${_cap}")
                if(_cap MATCHES "^[0-9]+$")
                    list(APPEND _detected "${_cap}")
                endif()
            endforeach()
            list(REMOVE_DUPLICATES _detected)
        endif()
    endif()

    if(_detected)
        set(CMAKE_CUDA_ARCHITECTURES ${_detected})
        message(STATUS "CUDA_ARCH=native -> detected sm_${_detected}")
    else()
        set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89)
        message(WARNING
            "CUDA_ARCH=native but no GPU was detected at configure time.\n"
            "Falling back to the portable list (sm_70/75/80/86/89). Pass\n"
            "-DCUDA_ARCH=<cc> (for example -DCUDA_ARCH=86) to target one GPU.")
    endif()
elseif(CUDA_ARCH STREQUAL "portable")
    # Turing through Ada. Add 90 for Hopper, 120 for Blackwell.
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89)
else()
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
endif()

project(cuda-learning
        VERSION 1.0.0
        DESCRIPTION "A hands-on CUDA curriculum, from C++ fundamentals to GPU mastery"
        LANGUAGES CXX CUDA)

# ------------------------------------------------------------
# Language standards and flags
# ------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generates compile_commands.json for clangd / VS Code IntelliSense.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
# -G disables most device optimisations and makes kernels much
# slower, so never benchmark a Debug build.
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -lineinfo")
# -lineinfo keeps the source-line mapping that Nsight Compute needs
# to attribute stalls to lines, at no runtime cost.

# Show register and shared-memory usage per kernel. This is the
# cheapest occupancy diagnostic there is, so it is on by default.
option(CL_PTXAS_VERBOSE "Print per-kernel register/shared usage" OFF)
if(CL_PTXAS_VERBOSE)
    string(APPEND CMAKE_CUDA_FLAGS " --ptxas-options=-v")
endif()

option(CL_ENABLE_OPTIONAL
       "Build exercises that need external dependencies (CUTLASS, OpenGL, ...)" OFF)

find_package(CUDAToolkit REQUIRED)

# Optional: the Phase 1 CPU baselines use OpenMP to show what
# multi-core parallelism buys before we move to the GPU. Exercises
# guard the code with #ifdef _OPENMP, so a missing OpenMP is fine.
find_package(OpenMP COMPONENTS CXX)

# ------------------------------------------------------------
# Exercise discovery
# ------------------------------------------------------------
set(CL_COMMON_DIR "${CMAKE_SOURCE_DIR}/common")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CudaLearning)

enable_testing()
# Some exercises write images or reports; give them a scratch dir
# so they never litter the source tree.
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/testrun")

message(STATUS "")
message(STATUS "Discovering exercises...")
cl_discover_exercises()

# ------------------------------------------------------------
# Convenience targets
# ------------------------------------------------------------
get_property(CL_SOLUTIONS GLOBAL PROPERTY CL_SOLUTION_TARGETS)
get_property(CL_STARTERS GLOBAL PROPERTY CL_STARTER_TARGETS)
get_property(CL_ALL GLOBAL PROPERTY CL_ALL_EXERCISES)
get_property(CL_SKIPPED GLOBAL PROPERTY CL_SKIPPED_EXERCISES)

list(LENGTH CL_ALL CL_COUNT)
list(LENGTH CL_SOLUTIONS CL_SOL_COUNT)
list(LENGTH CL_STARTERS CL_START_COUNT)

# `cmake --build build --target solutions` builds every reference
# solution and nothing else - the fastest way to check the repo.
if(CL_SOLUTIONS)
    add_custom_target(solutions DEPENDS ${CL_SOLUTIONS})
endif()
if(CL_STARTERS)
    add_custom_target(starters DEPENDS ${CL_STARTERS})
endif()

message(STATUS "")
message(STATUS "============================================================")
message(STATUS "  CUDA Learning ${PROJECT_VERSION}")
message(STATUS "------------------------------------------------------------")
message(STATUS "  CUDA toolkit    : ${CUDAToolkit_VERSION}")
message(STATUS "  Architectures   : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build type      : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Exercises found : ${CL_COUNT} (${CL_START_COUNT} starters, "
               "${CL_SOL_COUNT} solutions)")
if(CL_SKIPPED)
    message(STATUS "  Skipped         : ${CL_SKIPPED}")
    message(STATUS "                    (configure with -DCL_ENABLE_OPTIONAL=ON to try)")
endif()
message(STATUS "------------------------------------------------------------")
message(STATUS "  Build all        : cmake --build build -j")
message(STATUS "  Build solutions  : cmake --build build --target solutions -j")
message(STATUS "  Run the suite    : ctest --test-dir build --output-on-failure")
message(STATUS "  Run one phase    : ctest --test-dir build -L p3")
message(STATUS "============================================================")
