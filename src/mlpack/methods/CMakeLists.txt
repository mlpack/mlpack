cmake_minimum_required(VERSION 2.8)

## recurse
set(DIRS
  emst
  gmm
  hmm
  kmeans
  lars
  linear_regression
  mvu
  naive_bayes
  nca
  neighbor_search
  pca
  radical
  range_search
)

foreach(dir ${DIRS})
    add_subdirectory(${dir})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
