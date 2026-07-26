# Link OpenMP when the toolchain has it. The sources guard every
# OpenMP use with #ifdef _OPENMP, so a build without it still works
# (it just skips the multi-threaded variant).
if(OpenMP_CXX_FOUND)
    set(EX_LIBS OpenMP::OpenMP_CXX)
endif()

# The naive baseline at n=768 takes a few seconds, and the tile
# sweep runs five more multiplications on top.
set(EX_TIMEOUT 600)
