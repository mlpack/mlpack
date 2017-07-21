# mlpack test executable.
add_executable(mlpack_test
  mlpack_test.cpp
  rbm_network_test.cpp
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
add_custom_command(TARGET mlpack_test
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E tar xjpf digits_train.tar.bz2
  COMMAND ${CMAKE_COMMAND} -E tar xjpf digits_test.tar.bz2
  COMMAND ${CMAKE_COMMAND} -E tar xjpf digits_train_label.tar.bz2
  COMMAND ${CMAKE_COMMAND} -E tar xjpf digits_test_label.tar.bz2
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)

# The list of long running parallel tests
set(parallel_tests
  "SdpPrimalDualTest;SVDIncrementalTest;SVDBatchTest;"
  "LocalCoordinateCodingTest;FeedForwardNetworkTest;SparseAutoencoderTest;"
  "GMMTest;CFTest;ConvolutionalNetworkTest;HMMTest;LARSTest;"
  "LogisticRegressionTest")

# Add tests to the testing framework
# Get the list of sources from the test target
get_target_property(test_sources mlpack_test SOURCES)

# Go through the list of test sources and parse the test suite name
foreach(test_file ${test_sources})
  # Regex for parsing files with AUTO_TEST_SUITE
  file(STRINGS ${test_file} test_suite REGEX "BOOST_AUTO_TEST_SUITE\\(.*")
  if(NOT "${test_suite}" STREQUAL "")
    # Get the substring of test_suite within brackets in test_name
    string(REGEX MATCH "\\(.*\\)" test_name ${test_suite})
    # Get the substring excluding the brackets, by calculating the indices
    string(LENGTH ${test_name} end_idx)
    math(EXPR end_idx "${end_idx} - 2")
    string(SUBSTRING ${test_name} "1" ${end_idx} test)
    # Add the test to the testing tool, test is the name of the test suite
    add_test(NAME ${test} COMMAND mlpack_test -t ${test} WORKING_DIRECTORY
      ${CMAKE_BINARY_DIR})
  endif()
  # Regex for parsing files with FIXTURE_TEST_SUITE similarly
  file(STRINGS ${test_file} test_suite REGEX "BOOST_FIXTURE_TEST_SUITE\\(.*,")
  if(NOT "${test_suite}" STREQUAL "")
    # Get the substring of test_suite within brackets and comma in test_name
    string(REGEX MATCH "\\(.*," test_name ${test_suite})
    string(LENGTH ${test_name} end_idx)
    math(EXPR end_idx "${end_idx} - 2")
    string(SUBSTRING ${test_name} "1" ${end_idx} test)
    add_test(NAME ${test} COMMAND mlpack_test -t ${test} WORKING_DIRECTORY
      ${CMAKE_BINARY_DIR})
  endif()
endforeach()

# Use RUN_SERIAL for long running parallel tests
# set_tests_properties(${parallel_tests} PROPERTIES RUN_SERIAL TRUE)
