cmake_minimum_required(VERSION 2.8)

# we just want to recurse into the child directories here
set(DIRS
   clusterwise_regression
#   fast_multipole_method  ## awaiting resolution of #10 (libint dependency)
   gp_regression
#   kde  ## does not compile
   linear_regression  
#   lprcde  ## does not compile (missing NWRCdeProblem<>::MultiTreeQuerySummary::PostAccumulate)
#   matrix_factorized_fmm  ## does not compile right
   multigrid
   multitree_template  ## requires memory manager (mmapmm)
   proximity_project  ## requires memorymanager (mmapmm)
   regression
)

foreach(dir ${DIRS})
   add_subdirectory(${dir})
endforeach()

set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} PARENT_SCOPE)
