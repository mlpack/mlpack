cmake_minimum_required(VERSION 2.8)

# we just want to recurse into the child directories here
set(DIRS
   compression
#   fast_multipole_method  ## awaiting resolution of #10 (libint dependency)
#   kde  ## requires memory manager (mmapmm)
   linear_algebra
   linear_regression  ## requires memory manager (mmapmm)
#   lprcde  ## requires memory manager (mmapmm)
#   matrix_factorized_fmm  ## does not compile right
#   multitree_template  ## requires memory manager (mmapmm)
   nested_summation_template
   pca
#   proximity_project  ## requires memorymanager (mmapmm)
#   regression
#   thor_kde  ## does not compile: linker error (investigate)
)

foreach(dir ${DIRS})
   add_subdirectory(${dir})
endforeach()

set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} PARENT_SCOPE)
