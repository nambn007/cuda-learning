# ============================================================
# CMakeLists.txt – CUDA Learning Project
# ============================================================
# Usage:
#   mkdir build && cd build
#   cmake .. -DCMAKE_CUDA_ARCHITECTURES=75
#   make -j$(nproc)
# ============================================================

cmake_minimum_required(VERSION 3.18)
project(cuda-learning LANGUAGES CXX CUDA)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default CUDA architecture (GTX 1660 SUPER = sm_75)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0 --ptxas-options=-v")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --ptxas-options=-v")

# Enable separable compilation for dynamic parallelism
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# ============================================================
# Output directories
# ============================================================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ============================================================
# Helper function to add CUDA exercises
# ============================================================
function(add_cuda_exercise EXERCISE_NAME SOURCE_FILES)
    add_executable(${EXERCISE_NAME} ${SOURCE_FILES} ${ARGN})
    set_target_properties(${EXERCISE_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_include_directories(${EXERCISE_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/common)
endfunction()

# ============================================================
# Common utilities (shared across exercises)
# ============================================================
# Will contain timer, error checking macros, etc.
# add_library(common INTERFACE)
# target_include_directories(common INTERFACE common/)

# ============================================================
# Phase 2: CUDA Fundamentals
# ============================================================
# Uncomment as you complete exercises:

# add_cuda_exercise(hello_cuda phase-2-cuda-fundamentals/exercises/01-hello-cuda/main.cu)
# add_cuda_exercise(vector_add phase-2-cuda-fundamentals/exercises/02-vector-add/main.cu)
# add_cuda_exercise(saxpy phase-2-cuda-fundamentals/exercises/03-saxpy/main.cu)
# add_cuda_exercise(matrix_multiply phase-2-cuda-fundamentals/exercises/04-matrix-multiply/main.cu)

# ============================================================
# Phase 3: Intermediate
# ============================================================
# add_cuda_exercise(tiled_matmul phase-3-intermediate/exercises/01-shared-memory-matmul/main.cu)
# add_cuda_exercise(reduction phase-3-intermediate/exercises/02-parallel-reduction/main.cu)
# add_cuda_exercise(prefix_sum phase-3-intermediate/exercises/03-prefix-sum/main.cu)
# add_cuda_exercise(convolution_2d phase-3-intermediate/exercises/04-convolution-2d/main.cu)

# ============================================================
# Phase 4: Advanced
# ============================================================
# add_cuda_exercise(streams_pipeline phase-4-advanced/exercises/01-streams-pipeline/main.cu)
# add_cuda_exercise(cuda_graphs phase-4-advanced/exercises/02-cuda-graphs/main.cu)

# ============================================================
# Phase 5: Expert
# ============================================================
# add_cuda_exercise(cublas_bench phase-5-expert/exercises/01-cublas-benchmark/main.cu)
# target_link_libraries(cublas_bench PRIVATE cublas)
# add_cuda_exercise(tensor_gemm phase-5-expert/exercises/02-tensor-core-gemm/main.cu)

message(STATUS "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
message(STATUS "  CUDA Learning Project")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
