/**
 * @file sgd_test.cpp
 * @author Ryan Curtin
 *
 * Test file for SGD (stochastic gradient descent).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/cdk/cdk.hpp>
#include <mlpack/core/optimizers/sgd/test_function.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

BOOST_AUTO_TEST_SUITE(CDKTest);

BOOST_AUTO_TEST_CASE(SimpleCDKTestFunction)
{
  SGDTestFunction f;
  CDK<SGDTestFunction> cdk(f, 5, 1e-9, 5000000, true, true);

  arma::mat coordinates = f.GetInitialPoint();
  cdk.Optimize(coordinates);
}

BOOST_AUTO_TEST_SUITE_END();
