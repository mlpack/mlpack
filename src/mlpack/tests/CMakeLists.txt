cmake_minimum_required(VERSION 2.8)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all MLPACK sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# test executable
add_executable(mlpack_test
  mlpack_test.cpp
  allkfn_test.cpp
  allknn_test.cpp
  cli_test.cpp
  emst_test.cpp
#  hmm_test.cpp
  infomax_ica_test.cpp
  kernel_test.cpp
  lin_alg_test.cpp
  linear_regression_test.cpp
  load_save_test.cpp
  math_test.cpp
  nbc_test.cpp
  nca_test.cpp
  nnsvm_test.cpp
  ridge_regression_test.cpp
  save_restore_utility_test.cpp
  sort_policy_test.cpp
  svm_test.cpp
  tree_test.cpp
  union_find_test.cpp
)
# link dependencies of test executable
target_link_libraries(mlpack_test
  mlpack
  boost_unit_test_framework
)
