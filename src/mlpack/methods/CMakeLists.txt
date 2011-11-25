cmake_minimum_required(VERSION 2.8)

## recurse
set(DIRS
  emst
  fastica
  #gmm
  hmm
  infomax_ica
  kmeans
  linear_regression
  #mvu  # (currently known to not work)
  naive_bayes
  nca
  neighbor_search
  pca
)

foreach(dir ${DIRS})
    add_subdirectory(${dir})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
