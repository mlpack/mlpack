cmake_minimum_required(VERSION 2.8)

# MLPACK test executable.
add_executable(mlpack_test
  EXCLUDE_FROM_ALL
  mlpack_test.cpp
  allkfn_test.cpp
  allknn_test.cpp
  cli_test.cpp
  distribution_test.cpp
  emst_test.cpp
  gmm_test.cpp
  hmm_test.cpp
  kernel_test.cpp
  kmeans_test.cpp
  lars_test.cpp
  lin_alg_test.cpp
  linear_regression_test.cpp
  load_save_test.cpp
  math_test.cpp
  nbc_test.cpp
  nca_test.cpp
  pca_test.cpp
  radical_test.cpp
  range_search_test.cpp
  save_restore_utility_test.cpp
  sort_policy_test.cpp
  tree_test.cpp
  union_find_test.cpp
)
# Link dependencies of test executable.
target_link_libraries(mlpack_test
  mlpack
  boost_unit_test_framework
)

# Copy test data into right place.
add_custom_command(TARGET mlpack_test
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/data/ ${PROJECT_BINARY_DIR}
)
