# mlpack test executable.
add_executable(mlpack_test
  mlpack_test.cpp
  adaboost_test.cpp
  adam_test.cpp
  ada_delta_test.cpp
  allkrann_search_test.cpp
  arma_extend_test.cpp
  aug_lagrangian_test.cpp
  cf_test.cpp
  cli_test.cpp
  cosine_tree_test.cpp
  decision_stump_test.cpp
  det_test.cpp
  distribution_test.cpp
  emst_test.cpp
  fastmks_test.cpp
  gmm_test.cpp
  hmm_test.cpp
  hoeffding_tree_test.cpp
  ind2sub_test.cpp
  kernel_test.cpp
  kernel_pca_test.cpp
  kernel_traits_test.cpp
  kfn_test.cpp
  kmeans_test.cpp
  knn_test.cpp
  lars_test.cpp
  lbfgs_test.cpp
  lin_alg_test.cpp
  linear_regression_test.cpp
  load_save_test.cpp
  local_coordinate_coding_test.cpp
  log_test.cpp
  logistic_regression_test.cpp
  lrsdp_test.cpp
  lsh_test.cpp
  math_test.cpp
  matrix_completion_test.cpp
  maximal_inputs_test.cpp
  mean_shift_test.cpp
  metric_test.cpp
  minibatch_sgd_test.cpp
  nbc_test.cpp
  nca_test.cpp
  network_util_test.cpp
  nmf_test.cpp
  pca_test.cpp
  perceptron_test.cpp
  quic_svd_test.cpp
  radical_test.cpp
  range_search_test.cpp
  rectangle_tree_test.cpp
  regularized_svd_test.cpp
  sa_test.cpp
  sdp_primal_dual_test.cpp
  sgd_test.cpp
  serialization.hpp
  serialization.cpp
  serialization_test.cpp
  softmax_regression_test.cpp
  sort_policy_test.cpp
  sparse_autoencoder_test.cpp
  sparse_coding_test.cpp
  split_data_test.cpp
  termination_policy_test.cpp
  tree_test.cpp
  tree_traits_test.cpp
  union_find_test.cpp
  svd_batch_test.cpp
  svd_incremental_test.cpp
  nystroem_method_test.cpp
  armadillo_svd_test.cpp
  recurrent_network_test.cpp
)
# Link dependencies of test executable.
target_link_libraries(mlpack_test
  mlpack
  ${BOOST_unit_test_framework_LIBRARY}
)

# Copy test data into right place.
add_custom_command(TARGET mlpack_test
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/data/
      ${PROJECT_BINARY_DIR}
)
add_custom_command(TARGET mlpack_test
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E tar xjpf mnist_first250_training_4s_and_9s.tar.bz2
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)

# For 'make test'.
add_test (NAME mlpack_test
  COMMAND mlpack_test "--log_level=test_suite" # Set UTF runtime param
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/ # This is where test files are put.
)
