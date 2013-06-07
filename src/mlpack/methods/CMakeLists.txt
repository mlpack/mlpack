## Recurse into different methods.
set(DIRS
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
  mvu
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
# Density estimation trees fail on Windows.
if (NOT WIN32)
  set(DIRS ${DIRS} det)
endif (NOT WIN32)

foreach(dir ${DIRS})
    add_subdirectory(${dir})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
