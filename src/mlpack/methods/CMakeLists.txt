# Recurse into each method mlpack provides.
set(DIRS
  cf
  det
  emst
  fastmks
  gmm
  hmm
  kernel_pca
  kmeans
  lars
  linear_regression
  local_coordinate_coding
  lsh
  naive_bayes
  nca
  neighbor_search
  nmf
  pca
  radical
  range_search
  rann
  sparse_coding
)

foreach(dir ${DIRS})
    add_subdirectory(${dir})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
