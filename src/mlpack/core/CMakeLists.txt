# All we have to do is recurse into the subdirectories.
set(DIRS
  arma_extend
  boost_backport
  data
  dists
  kernels
  math
  metrics
  optimizers
  tree
  util
)

foreach(dir ${DIRS})
  add_subdirectory(${dir})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} PARENT_SCOPE)
