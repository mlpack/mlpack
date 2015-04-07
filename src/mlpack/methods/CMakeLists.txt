# Recurse into each method mlpack provides.
set(DIRS
  adaboost 
  amf
  cf
  decision_stump
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
  logistic_regression
  lsh
#  mvu
  matrix_completion
  naive_bayes
  nca
  neighbor_search
  nmf
#  lmf
  manifold
  pca
  perceptron
  quic_svd
  radical
  range_search
  rann
  regularized_svd
  softmax_regression
  sparse_autoencoder
  sparse_coding
  nystroem_method
)

foreach(dir ${DIRS})
    add_subdirectory(${dir})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
