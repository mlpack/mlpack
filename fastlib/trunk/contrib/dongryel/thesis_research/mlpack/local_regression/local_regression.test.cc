/** @file local_regression.test.cc
 *
 *  A "stress" test driver for local regression.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

// for BOOST testing
#define BOOST_TEST_MAIN

#include "mlpack/local_regression/test_local_regression.h"
#include <time.h>

namespace mlpack {
namespace local_regression {
namespace test_local_regression {
int num_dimensions_;
int num_points_;
}
}
}

BOOST_AUTO_TEST_SUITE(TestSuiteLocalRegression)
BOOST_AUTO_TEST_CASE(TestCaseLocalRegression) {

  // Call the tests.
  mlpack::local_regression::TestLocalRegression local_regression_test;
  local_regression_test.StressTestMain();

  std::cout << "All tests passed!\n";
}
BOOST_AUTO_TEST_SUITE_END()
