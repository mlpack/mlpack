# All we have to do is recurse into the subdirectories.
set(DIRS
  arma_extend
  boost_backport
  cv
  data
  dists
  hpt
  kernels
  math
  metrics
  tree
  util
)

foreach(dir ${DIRS})
  add_subdirectory(${dir})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} PARENT_SCOPE)
