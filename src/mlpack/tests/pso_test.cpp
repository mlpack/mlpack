/**
 * @file pso_test.cpp
 * @author Chintan Soni
 *
 * Test file for PSO optimizer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/pso/pso.hpp>

#include <mlpack/core/optimizers/problems/rosenbrock_function.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

BOOST_AUTO_TEST_SUITE(PSOTest);

BOOST_AUTO_TEST_CASE(SimplePSOTest)
{
  RosenbrockFunction f;
  PSO s;

  arma::vec coordinates = f.GetInitialPoint();
  double result = s.Optimize(f, coordinates);

  BOOST_REQUIRE_SMALL(result, 1e-4);
}

BOOST_AUTO_TEST_SUITE_END();
