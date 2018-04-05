/**
 * @file spsa_test.cpp
 * @author N Rajiv Vaidyanathan
 *
 * Test file for SPSA (simultaneous pertubation stochastic approximation).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/spsa/spsa.hpp>
#include <mlpack/core/optimizers/problems/sgd_test_function.hpp>
#include <mlpack/core/optimizers/problems/generalized_rosenbrock_function.hpp>
#include <mlpack/core/optimizers/problems/sphere_function.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression_function.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;
using namespace mlpack::regression;

BOOST_AUTO_TEST_SUITE(SPSATest);

BOOST_AUTO_TEST_CASE(SimpleSPSATestFunction)
{
  for (size_t i = 10; i <= 50; i++)
  {
	  SphereFunction h(i);
	  GeneralizedRosenbrockFunction f(i);

	  SPSA optimiser(0.602, 0.101, 1e-6,
	                 0.01, 100000);

	  arma::mat coordinates = f.GetInitialPoint();
	  double result = optimiser.Optimize(h, coordinates);

	  BOOST_REQUIRE_CLOSE(result, 12, 5);
	  BOOST_REQUIRE_SMALL(coordinates[0], 1.0);
	  BOOST_REQUIRE_SMALL(coordinates[1], 1.0);
	  BOOST_REQUIRE_SMALL(coordinates[2], 1.0);
  }
}
BOOST_AUTO_TEST_SUITE_END();
