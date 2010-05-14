cmake_minimum_required(VERSION 2.8)

# we just want to recurse into the child directories here
set(DIRS
#   compression
#   fast_multipole_method  ## awaiting resolution of #10 (libint dependency)
#   kde  ## does not compile
   linear_algebra
   linear_regression  
#   lprcde  ## does not compile (missing NWRCdeProblem<>::MultiTreeQuerySummary::PostAccumulate)
#   matrix_factorized_fmm  ## does not compile right
   multitree_template  ## requires memory manager (mmapmm)
#   nested_summation_template
#   pca
   proximity_project  ## requires memorymanager (mmapmm)
   regression
#   thor_kde  ## does not compile: linker error (investigate)
)

foreach(dir ${DIRS})
   add_subdirectory(${dir})
endforeach()

set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} PARENT_SCOPE)
