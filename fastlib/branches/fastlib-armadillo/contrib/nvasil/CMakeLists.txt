cmake_minimum_required(VERSION 2.8)

# we just want to recurse into the child directories here
set(DIRS
#   all_centroid_knn  ## does not compile: fastlib::fx has changed
#   allkfn  ## this exists in MLPACK non-contrib already
#   convex_nmf  ## depends on l_bfgs code which does not exist
#   dual_manifold  ## depends on l_bfgs code which does not exist
#   l_bfgs  ## directory is empty
#   manifold_plot    ## MATLAB routines
#   multiscale_mvu  ## depends on l_bfgs code which does not exist
#   mvu  ## depends on l_bfgs code which does not exist
   nmf_tree
#   non_convex_mvu  ## fastlib::fx syntax has changed, does not compile
#   non_convex_nmf  ## depends on l_bfgs code which does not exist
#   test_svd   ## directory is empty
)

foreach(dir ${DIRS})
   add_subdirectory(${dir})
endforeach()

set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} PARENT_SCOPE)
