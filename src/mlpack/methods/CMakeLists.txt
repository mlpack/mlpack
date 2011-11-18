cmake_minimum_required(VERSION 2.8)

## recurse
set(DIRS
  emst
  fastica
  gmm
  hmm
  infomax_ica
  # kernel_pca # (required sparse and is known to not work or compile)
  linear_regression
  #mvu  # (currently known to not work)
  naive_bayes
  nca
  pca
  neighbor_search
  nnsvm
  regression
  svm
)

foreach(dir ${DIRS})
    add_subdirectory(${dir})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
