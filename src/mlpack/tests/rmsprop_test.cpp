/**
 * @file rmsprop_test.cpp
 * @author Marcus Edel
 *
 * Tests the RMSProp optimizer.
 */
#include <mlpack/core.hpp>

#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>
#include <mlpack/core/optimizers/lbfgs/test_functions.hpp>
#include <mlpack/core/optimizers/sgd/test_function.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

BOOST_AUTO_TEST_SUITE(RMSpropTest);

/**
 * Tests the RMSprop optimizer using a simple test function.
 */
BOOST_AUTO_TEST_CASE(SimpleRMSpropTestFunction)
{
  SGDTestFunction f;
  RMSprop<SGDTestFunction> optimizer(f, 1e-3, 0.99, 1e-8, 5000000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  double result = optimizer.Optimize(coordinates);

  BOOST_REQUIRE_CLOSE(result, -1.0, 0.05);
  BOOST_REQUIRE_SMALL(coordinates[0], 1e-3);
  BOOST_REQUIRE_SMALL(coordinates[1], 1e-3);
  BOOST_REQUIRE_SMALL(coordinates[2], 1e-3);
}

BOOST_AUTO_TEST_SUITE_END();
