/** @file kde.test.cc
 *
 *  A "stress" test driver for KDE.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

// for BOOST testing
#define BOOST_TEST_MAIN

#include "mlpack/kde/test_kde.h"
#include <time.h>

namespace mlpack {
namespace kde {
namespace test_kde {
int num_dimensions_;
int num_points_;
}
}
}

BOOST_AUTO_TEST_SUITE(TestSuiteKde)
BOOST_AUTO_TEST_CASE(TestCaseKde) {

  // Call the tests.
  mlpack::kde::TestKde kde_test;
  kde_test.StressTestMain();

  std::cout << "All tests passed!\n";
}
BOOST_AUTO_TEST_SUITE_END()
