include(CTest)

# mlpack test executable.
add_executable(mlpack_test
  ${MLPACK_TEST_SRCS}

  # Tests for neural network code (and reinforcement learning code).
  ann/activation_functions_test.cpp
  ann/ann_dist_test.cpp
  # This includes all files in ann/layer/.
  ann/layer_test.cpp
  ann/ann_regularizer_test.cpp
  ann/ann_test_tools.hpp
  ann/convolutional_network_test.cpp
  ann/convolution_test.cpp
  ann/custom_layer.hpp
  ann/feedforward_network_test.cpp
  ann/init_rules_test.cpp
  ann/ksinit_test.cpp
  ann/loss_functions_test.cpp
  ann/recurrent_network_test.cpp

  ann/async_learning_test.cpp
  ann/augmented_rnns_tasks_test.cpp
  ann/reward_clipping_test.cpp
  ann/rl_components_test.cpp

  # These tests have not yet been adapted to the new inheritance-based ANN
  # implementation.
#  ann/not_adapted/activation_functions_test.cpp
#  ann/not_adapted/ann_layer_test.cpp
#  ann/not_adapted/feedforward_network_test.cpp
#  ann/not_adapted/recurrent_network_test.cpp
#  ann/not_adapted/rbm_network_test.cpp
#  ann/not_adapted/rnn_reber_test.cpp
#  ann/not_adapted/dcgan_test.cpp
#  ann/not_adapted/gan_test.cpp
#  ann/not_adapted/wgan_test.cpp

  # Tests for other methods.
  adaboost_test.cpp
  akfn_test.cpp
  aknn_test.cpp
  armadillo_svd_test.cpp
  arma_extend_test.cpp
  bayesian_linear_regression_test.cpp
  bias_svd_test.cpp
  binarize_test.cpp
  block_krylov_svd_test.cpp
  callback_test.cpp
  cf_test.cpp
  cli_binding_test.cpp
  cosine_tree_test.cpp
  cv_test.cpp
  dbscan_test.cpp
  decision_tree_regressor_test.cpp
  decision_tree_test.cpp
  det_test.cpp
  digamma_test.cpp
  distribution_test.cpp
  drusilla_select_test.cpp
  emst_test.cpp
  facilities_test.cpp
  fastmks_test.cpp
  gmm_test.cpp
  hmm_test.cpp
  hpt_test.cpp
  hoeffding_tree_test.cpp
  hyperplane_test.cpp
  image_load_test.cpp
  imputation_test.cpp
  io_test.cpp
  kde_test.cpp
  kernel_pca_test.cpp
  kernel_test.cpp
  kernel_traits_test.cpp
  kfn_test.cpp
  kmeans_test.cpp
  knn_test.cpp
  krann_search_test.cpp
  lars_test.cpp
  lin_alg_test.cpp
  linear_regression_test.cpp
  lmnn_test.cpp
  linear_svm_test.cpp
  load_save_test.cpp
  local_coordinate_coding_test.cpp
  logistic_regression_test.cpp
  log_test.cpp
  lsh_test.cpp
  main.cpp
  math_test.cpp
  matrix_completion_test.cpp
  maximal_inputs_test.cpp
  metric_test.cpp
  mean_shift_test.cpp
  mock_categorical_data.hpp
  nbc_test.cpp
  nca_test.cpp
  nmf_test.cpp
  nystroem_method_test.cpp
  octree_test.cpp
  one_hot_encoding_test.cpp
  pca_test.cpp
  perceptron_test.cpp
  prefixedoutstream_test.cpp
  python_binding_test.cpp
  qdafn_test.cpp
  quic_svd_test.cpp
  q_learning_test.cpp
  radical_test.cpp
  random_forest_test.cpp
  random_test.cpp
  randomized_svd_test.cpp
  range_search_test.cpp
  rectangle_tree_test.cpp
  regularized_svd_test.cpp
  scaling_test.cpp
  size_checks_test.cpp
  serialization.cpp
  serialization.hpp
  serialization_test.cpp
  sfinae_test.cpp
  softmax_regression_test.cpp
  sort_policy_test.cpp
  sparse_autoencoder_test.cpp
  sparse_coding_test.cpp
  spill_tree_test.cpp
  split_data_test.cpp
  string_encoding_test.cpp
  sumtree_test.cpp
  svd_batch_test.cpp
  svd_incremental_test.cpp
  svdplusplus_test.cpp
  termination_policy_test.cpp
  test_catch_tools.hpp
  test_function_tools.hpp
  timer_test.cpp
  tree_test.cpp
  tree_traits_test.cpp
  trigamma_test.cpp
  ub_tree_test.cpp
  union_find_test.cpp
  vantage_point_tree_test.cpp
  xgboost_test.cpp

  # Tests for individual bindings.
  main_tests/adaboost_test.cpp
  main_tests/adaboost_train_test.cpp
  main_tests/adaboost_classify_test.cpp
  main_tests/adaboost_probabilities_test.cpp
  main_tests/approx_kfn_test.cpp
  main_tests/bayesian_linear_regression_test.cpp
  main_tests/cf_test.cpp
  main_tests/dbscan_test.cpp
  main_tests/decision_tree_test.cpp
  main_tests/det_test.cpp
  main_tests/emst_test.cpp
  main_tests/fastmks_test.cpp
  main_tests/gmm_generate_test.cpp
  main_tests/gmm_probability_test.cpp
  main_tests/gmm_train_test.cpp
  main_tests/hmm_generate_test.cpp
  main_tests/hmm_loglik_test.cpp
  main_tests/hmm_test_utils.hpp
  main_tests/hmm_train_test.cpp
  main_tests/hmm_viterbi_test.cpp
  main_tests/hoeffding_tree_test.cpp
  main_tests/image_converter_test.cpp
  main_tests/kde_test.cpp
  main_tests/kernel_pca_test.cpp
  main_tests/kfn_test.cpp
  main_tests/kmeans_test.cpp
  main_tests/knn_test.cpp
  main_tests/krann_test.cpp
  main_tests/linear_regression_test.cpp
  main_tests/linear_regression_train_test.cpp
  main_tests/linear_regression_predict_test.cpp
  main_tests/linear_svm_test.cpp
  main_tests/lmnn_test.cpp
  main_tests/local_coordinate_coding_test.cpp
  main_tests/logistic_regression_test.cpp
  main_tests/lsh_test.cpp
  main_tests/mean_shift_test.cpp
  main_tests/nbc_test.cpp
  main_tests/nca_test.cpp
  main_tests/nmf_test.cpp
  main_tests/pca_test.cpp
  main_tests/perceptron_test.cpp
  main_tests/preprocess_binarize_test.cpp
  main_tests/preprocess_imputer_test.cpp
  main_tests/preprocess_one_hot_encode_test.cpp
  main_tests/preprocess_scale_test.cpp
  main_tests/preprocess_split_test.cpp
  main_tests/radical_test.cpp
  main_tests/random_forest_test.cpp
  main_tests/range_search_test.cpp
  main_tests/softmax_regression_test.cpp
  main_tests/sparse_coding_test.cpp
  main_tests/main_test_fixture.hpp
)

if(NOT BUILD_SHARED_LIBS)
# Build mlpack test executable statically.
  target_link_libraries(mlpack_test -static
    ${MLPACK_LIBRARIES}
  )
else()
  # Build mlpack test executable dynamically.
  target_link_libraries(mlpack_test
    ${MLPACK_LIBRARIES}
  )
endif()

# If the user asked for running test cases with verbose output, turn that on.
if (TEST_VERBOSE)
  target_compile_definitions(mlpack_test PUBLIC
      -DMLPACK_PRINT_DEBUG -DMLPACK_PRINT_INFO -DMLPACK_PRINT_WARN)
else()
  # Otherwise, suppress all [FATAL] output.
  target_compile_definitions(mlpack_test PUBLIC -DMLPACK_SUPPRESS_FATAL)
endif()

set_target_properties(mlpack_test PROPERTIES COTIRE_CXX_PREFIX_HEADER_INIT "../core.hpp")
cotire(mlpack_test)

# Copy test data into right place.
add_custom_command(TARGET mlpack_test
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/data/
      ${PROJECT_BINARY_DIR}
)

add_custom_command(TARGET mlpack_test
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E tar xjf mnist_first250_training_4s_and_9s.tar.bz2
  COMMAND ${CMAKE_COMMAND} -E tar xjf digits_train.tar.bz2
  COMMAND ${CMAKE_COMMAND} -E tar xjf digits_test.tar.bz2
  COMMAND ${CMAKE_COMMAND} -E tar xjf digits_train_label.tar.bz2
  COMMAND ${CMAKE_COMMAND} -E tar xjf digits_test_label.tar.bz2
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)

add_test(NAME "catch_test" COMMAND mlpack_test WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

set_tests_properties("catch_test" PROPERTIES TIMEOUT 0)
