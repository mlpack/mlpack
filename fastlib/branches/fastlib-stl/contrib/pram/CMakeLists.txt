cmake_minimum_required(VERSION 2.8)

# we just want to recurse into the child directories here
set(DIRS
#   approx_nn  ## requires errno, does not compile (simple #include)
   covertrees
   density_tree
#   lle  ## fastlib::fx syntax has changed, does not compile
#   mog_l2e  ## linker error against la:: components, does not link
#   opt  ## fastlib::fx syntax has changed, does not compile
#   sparsepca  ## fastlib::fx syntax has changed, does not compile
)

foreach(dir ${DIRS})
   add_subdirectory(${dir})
endforeach()

set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} PARENT_SCOPE)
