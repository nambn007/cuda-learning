# The compute roof can only be measured with the instructions the
# machine actually has. Without -march=native the compiler targets
# baseline x86-64, which has no AVX and no FMA, and the "peak
# GFLOP/s" you measure is roughly 8x too low.
#
# The rest of the repository is deliberately built WITHOUT
# -march=native so the binaries stay portable. This one exercise is
# about the hardware limits, so here it is the right flag.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" CL_HAS_MARCH_NATIVE)
if(CL_HAS_MARCH_NATIVE)
    set(EX_COMPILE_OPTIONS -march=native)
endif()

set(EX_TIMEOUT 300)
