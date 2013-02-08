## recurse
set(DIRS
  det
  emst
#  fastmks
  gmm
  hmm
  kernel_pca
  kmeans
  lars
  linear_regression
#  lmnn
  local_coordinate_coding
  mvu
  naive_bayes
  nca
  neighbor_search
  nmf
  pca
  radical
  range_search
  sparse_coding
)

foreach(dir ${DIRS})
    add_subdirectory(${dir})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
