# mlpack test executable.
add_executable(mlpack_test
  main.cpp
  python_binding_test.cpp
  wgan_test.cpp
  main_tests/adaboost_test.cpp
  main_tests/approx_kfn_test.cpp
  main_tests/bayesian_linear_regression_test.cpp
  main_tests/cf_test.cpp
  main_tests/dbscan_test.cpp
  main_tests/decision_stump_test.cpp
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
  main_tests/lmnn_test.cpp
  main_tests/linear_svm_test.cpp
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
  main_tests/softmax_regression_test.cpp
  main_tests/sparse_coding_test.cpp
  main_tests/range_search_test.cpp
  main_tests/test_helper.hpp
)

# Link dependencies of test executable.
target_link_libraries(mlpack_test
  mlpack
  ${ARMADILLO_LIBRARIES}
  ${BOOST_LIBRARIES}
  ${COMPILER_SUPPORT_LIBRARIES}
)

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
