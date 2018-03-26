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

#include <mlpack/core/optimizers/problems/rastrigin_function.hpp>
#include <mlpack/core/optimizers/problems/rosenbrock_function.hpp>
#include <mlpack/core/optimizers/problems/sphere_function.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

BOOST_AUTO_TEST_SUITE(PSOTest);

BOOST_AUTO_TEST_CASE(PSORastriginTest)
{
  RastriginFunction f(4);
  LBestPSO s(16, 2500);

  arma::vec coordinates = f.GetInitialPoint();
  double result = s.Optimize(f, coordinates);

  BOOST_REQUIRE_SMALL(result, 1e-5);
  for (size_t j = 0; j < 4; ++j)
    BOOST_REQUIRE_SMALL(coordinates[j], 1e-3);
}

BOOST_AUTO_TEST_CASE(PSORosenbrockTest)
{
  RosenbrockFunction f;
  LBestPSO s(16, 2500);

  arma::vec coordinates = f.GetInitialPoint();
  double result = s.Optimize(f, coordinates);

  BOOST_REQUIRE_SMALL(result, 1e-5);
  for (size_t j = 0; j < 2; ++j)
    BOOST_REQUIRE_CLOSE(coordinates[j], (double) 1.0, 1e-3);
}

BOOST_AUTO_TEST_CASE(PSOSphereTest)
{
  SphereFunction f(4);
  LBestPSO s(16, 2500);

  arma::vec coordinates = f.GetInitialPoint();
  double result = s.Optimize(f, coordinates);

  BOOST_REQUIRE_SMALL(result, 1e-5);
  for (size_t j = 0; j < 4; ++j)
    BOOST_REQUIRE_SMALL(coordinates[j], 1e-3);
}

BOOST_AUTO_TEST_SUITE_END();
