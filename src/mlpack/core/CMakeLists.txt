# All we have to do is recurse into the subdirectories.
set(DIRS
  arma_extend
  col
  file
  io
  kernels
  math
  optimizers
  tree
  utilities
)

foreach(dir ${DIRS})
  add_subdirectory(${dir})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} PARENT_SCOPE)
