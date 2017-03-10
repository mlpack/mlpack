/**
 * @file momentum_sgd_test.cpp
 * @author Ryan Curtin
 *
 * Test file for MomentumSGD (stochastic gradient descent with momentum updates).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/core/optimizers/sgd/update_policies/momentum_update.hpp>
#include <mlpack/core/optimizers/lbfgs/test_functions.hpp>
#include <mlpack/core/optimizers/sgd/test_function.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

BOOST_AUTO_TEST_SUITE(MomentumSGDTest);

BOOST_AUTO_TEST_CASE(SimpleMomentumSGDTestFunction)
{
  SGDTestFunction f;
  MomentumUpdate momentumUpdate(0.9);
  MomentumSGD<SGDTestFunction> s(f, 0.0003, 5000000, 1e-9, true, momentumUpdate);

  arma::mat coordinates = f.GetInitialPoint();
  double result = s.Optimize(coordinates);

  BOOST_REQUIRE_CLOSE(result, -1.0, 0.05);
  BOOST_REQUIRE_SMALL(coordinates[0], 1e-3);
  BOOST_REQUIRE_SMALL(coordinates[1], 1e-7);
  BOOST_REQUIRE_SMALL(coordinates[2], 1e-7);
}

BOOST_AUTO_TEST_SUITE_END();
