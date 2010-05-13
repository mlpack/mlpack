cmake_minimum_required(VERSION 2.8)

# we just want to recurse into the child directories here
set(DIRS
#   hyperkernel_kde  ## missing ichol.h
#   local_kernel_machines  ## does not compile, LocalKernelMachines<SVMLinearKernel>::RunLocalKernelMachines_() fails to compile
#   local_polynomial_density  ## does not compile.
)

foreach(dir ${DIRS})
   add_subdirectory(${dir})
endforeach()

set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} PARENT_SCOPE)
