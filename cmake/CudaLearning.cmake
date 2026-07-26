# ============================================================
# cmake/CudaLearning.cmake - exercise auto-discovery
# ============================================================
# Every directory matching
#
#     phase-<N>-*/exercises/<NN>-<slug>/
#     phase-<N>-*/projects/<slug>/
#
# is turned into build targets automatically. There is no central
# list to maintain: drop in a new folder with a main.cu and it
# builds on the next `cmake ..`.
#
# Naming
# ------
#   main.cu      -> target  pN_NN_slug        (starter, has TODOs)
#   solution.cu  -> target  pN_NN_slug_sol    (reference solution)
#   *.cu / *.cpp -> compiled into BOTH targets (shared kernels)
#
# Per-exercise overrides
# ----------------------
# Drop an `exercise.cmake` in the exercise folder to change the
# defaults. It is include()d before the targets are created:
#
#   set(EX_LIBS CUDA::cublas)          # extra libraries to link
#   set(EX_ENABLED OFF)                # do not build at all
#   set(EX_SKIP_REASON "needs CUTLASS")
#   set(EX_TEST OFF)                   # exclude from `ctest`
#   set(EX_ARGS --size 1024)           # arguments used by `ctest`
#   set(EX_SEPARABLE ON)               # relocatable device code
#   set(EX_TIMEOUT 300)                # ctest timeout, seconds
#   set(EX_DEFINITIONS FOO=1)          # extra -D flags
#   set(EX_CXX_ONLY ON)                # never treat sources as CUDA
# ============================================================

# phase-2-cuda-fundamentals -> p2
function(cl_phase_prefix PHASE_DIR OUT_VAR)
    string(REGEX MATCH "^phase-([0-9]+)" _matched "${PHASE_DIR}")
    if(_matched)
        set(${OUT_VAR} "p${CMAKE_MATCH_1}" PARENT_SCOPE)
    else()
        set(${OUT_VAR} "px" PARENT_SCOPE)
    endif()
endfunction()

# 03-vector-add -> 03_vector_add
function(cl_sanitize NAME OUT_VAR)
    string(REGEX REPLACE "[^A-Za-z0-9]" "_" _clean "${NAME}")
    set(${OUT_VAR} "${_clean}" PARENT_SCOPE)
endfunction()

# ------------------------------------------------------------
# cl_add_exercise - create the starter and solution targets
# ------------------------------------------------------------
function(cl_add_exercise PREFIX EX_DIR_NAME EX_PATH)
    # Defaults, possibly overridden by exercise.cmake below.
    set(EX_LIBS "")
    set(EX_ENABLED ON)
    set(EX_SKIP_REASON "")
    set(EX_TEST ON)
    set(EX_ARGS "")
    set(EX_SEPARABLE OFF)
    set(EX_TIMEOUT 180)
    set(EX_DEFINITIONS "")
    set(EX_COMPILE_OPTIONS "")
    set(EX_CXX_ONLY OFF)

    if(EXISTS "${EX_PATH}/exercise.cmake")
        include("${EX_PATH}/exercise.cmake")
    endif()

    cl_sanitize("${EX_DIR_NAME}" _slug)
    set(_base "${PREFIX}_${_slug}")

    if(NOT EX_ENABLED)
        message(STATUS "  [skip] ${_base} - ${EX_SKIP_REASON}")
        set_property(GLOBAL APPEND PROPERTY CL_SKIPPED_EXERCISES
                     "${_base} (${EX_SKIP_REASON})")
        return()
    endif()

    # Sources shared by both variants (helper kernels, host code).
    file(GLOB _shared "${EX_PATH}/*.cu" "${EX_PATH}/*.cpp")
    list(FILTER _shared EXCLUDE REGEX "/(main|solution)\\.(cu|cpp)$")

    set(_built_any FALSE)
    foreach(_variant IN ITEMS main solution)
        set(_entry "")
        foreach(_ext IN ITEMS cu cpp)
            if(EXISTS "${EX_PATH}/${_variant}.${_ext}")
                set(_entry "${EX_PATH}/${_variant}.${_ext}")
                break()
            endif()
        endforeach()
        if(NOT _entry)
            continue()
        endif()

        if(_variant STREQUAL "solution")
            set(_target "${_base}_sol")
        else()
            set(_target "${_base}")
        endif()

        add_executable(${_target} "${_entry}" ${_shared})
        target_include_directories(${_target} PRIVATE
            "${CL_COMMON_DIR}" "${EX_PATH}")
        target_compile_features(${_target} PRIVATE cxx_std_17)

        if(EX_DEFINITIONS)
            target_compile_definitions(${_target} PRIVATE ${EX_DEFINITIONS})
        endif()
        if(EX_COMPILE_OPTIONS)
            target_compile_options(${_target} PRIVATE ${EX_COMPILE_OPTIONS})
        endif()
        if(EX_LIBS)
            target_link_libraries(${_target} PRIVATE ${EX_LIBS})
        endif()

        # Relocatable device code is only needed for dynamic
        # parallelism and cross-translation-unit __device__ calls.
        # It costs compile time and blocks some optimisations, so
        # it stays off unless the exercise asks for it.
        if(EX_SEPARABLE)
            set_target_properties(${_target} PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_RESOLVE_DEVICE_SYMBOLS ON)
        endif()

        # Group binaries by phase so build/bin stays navigable.
        set_target_properties(${_target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/${PREFIX}")

        set(_built_any TRUE)
        if(_variant STREQUAL "solution")
            set_property(GLOBAL APPEND PROPERTY CL_SOLUTION_TARGETS ${_target})
            # Only solutions are run by ctest: a starter still full
            # of TODOs is *expected* to produce the wrong answer.
            if(EX_TEST)
                add_test(NAME ${_target}
                         COMMAND ${_target} ${EX_ARGS}
                         WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/testrun")
                set_tests_properties(${_target} PROPERTIES
                    TIMEOUT ${EX_TIMEOUT}
                    LABELS "${PREFIX}")
            endif()
        else()
            set_property(GLOBAL APPEND PROPERTY CL_STARTER_TARGETS ${_target})
        endif()
    endforeach()

    if(_built_any)
        set_property(GLOBAL APPEND PROPERTY CL_ALL_EXERCISES "${_base}")
    else()
        message(STATUS "  [empty] ${_base} - no main.cu/main.cpp or solution.cu/solution.cpp")
    endif()
endfunction()

# ------------------------------------------------------------
# cl_discover_exercises - walk the phase folders
# ------------------------------------------------------------
macro(cl_discover_exercises)
    file(GLOB _cl_phases RELATIVE "${CMAKE_SOURCE_DIR}" "${CMAKE_SOURCE_DIR}/phase-*")
    list(SORT _cl_phases)

    foreach(_cl_phase IN LISTS _cl_phases)
        if(IS_DIRECTORY "${CMAKE_SOURCE_DIR}/${_cl_phase}")
            cl_phase_prefix("${_cl_phase}" _cl_prefix)
            message(STATUS "${_cl_phase}")

            foreach(_cl_kind IN ITEMS exercises projects)
                set(_cl_root "${CMAKE_SOURCE_DIR}/${_cl_phase}/${_cl_kind}")
                if(NOT IS_DIRECTORY "${_cl_root}")
                    continue()
                endif()
                file(GLOB _cl_items RELATIVE "${_cl_root}" "${_cl_root}/*")
                list(SORT _cl_items)

                foreach(_cl_item IN LISTS _cl_items)
                    set(_cl_dir "${_cl_root}/${_cl_item}")
                    if(NOT IS_DIRECTORY "${_cl_dir}")
                        continue()
                    endif()
                    if(EXISTS "${_cl_dir}/CMakeLists.txt")
                        # The exercise brings its own build rules.
                        add_subdirectory("${_cl_dir}"
                            "${CMAKE_BINARY_DIR}/${_cl_phase}/${_cl_kind}/${_cl_item}")
                    else()
                        cl_add_exercise("${_cl_prefix}" "${_cl_item}" "${_cl_dir}")
                    endif()
                endforeach()
            endforeach()
        endif()
    endforeach()
endmacro()
