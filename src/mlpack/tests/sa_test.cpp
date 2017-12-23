/*
 * @file sa_test.cpp
 * @auther Zhihao Lou
 *
 * Test file for SA (simulated annealing).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/sa/sa.hpp>
#include <mlpack/core/optimizers/sa/exponential_schedule.hpp>
#include <mlpack/core/optimizers/problems/generalized_rosenbrock_function.hpp>
#include <mlpack/core/optimizers/problems/rosenbrock_function.hpp>
#include <mlpack/core/optimizers/problems/rastrigin_function.hpp>

#include <mlpack/core/metrics/ip_metric.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/core/metrics/mahalanobis_distance.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;
using namespace mlpack::metric;

BOOST_AUTO_TEST_SUITE(SATest);

// The Generalized-Rosenbrock function is a simple function to optimize.
BOOST_AUTO_TEST_CASE(GeneralizedRosenbrockTest)
{
  size_t dim = 10;
  GeneralizedRosenbrockFunction f(dim);

  double iteration = 0;
  double result = DBL_MAX;
  arma::mat coordinates;
  while (result > 1e-6)
  {
    ExponentialSchedule schedule;
    // The convergence is very sensitive to the choices of maxMove and initMove.
    SA<ExponentialSchedule> sa(schedule, 1000000, 1000., 1000, 100, 1e-10, 3,
        1.5, 0.5, 0.3);
    coordinates = f.GetInitialPoint();
    result = sa.Optimize(f, coordinates);
    ++iteration;

    BOOST_REQUIRE_LT(iteration, 4); // No more than three tries.
  }

  // 0.1% tolerance for each coordinate.
  BOOST_REQUIRE_SMALL(result, 1e-6);
  for (size_t j = 0; j < dim; ++j)
      BOOST_REQUIRE_CLOSE(coordinates[j], (double) 1.0, 0.1);
}

// The Rosenbrock function is a simple function to optimize.
BOOST_AUTO_TEST_CASE(RosenbrockTest)
{
  RosenbrockFunction f;
  ExponentialSchedule schedule;
  // The convergence is very sensitive to the choices of maxMove and initMove.
  SA<> sa(schedule, 1000000, 1000., 1000, 100, 1e-11, 3, 1.5, 0.3, 0.3);
  arma::mat coordinates = f.GetInitialPoint();

  const double result = sa.Optimize(f, coordinates);

  BOOST_REQUIRE_SMALL(result, 1e-5);
  BOOST_REQUIRE_CLOSE(coordinates[0], 1.0, 1e-2);
  BOOST_REQUIRE_CLOSE(coordinates[1], 1.0, 1e-2);
}

/**
 * The Rastrigrin function, a (not very) simple nonconvex function. It has very
 * many local minima, so finding the true global minimum is difficult.
 */
BOOST_AUTO_TEST_CASE(RastrigrinFunctionTest)
{
  // Simulated annealing isn't guaranteed to converge (except in very specific
  // situations).  If this works 1 of 4 times, I'm fine with that.  All I want
  // to know is that this implementation will escape from local minima.
  size_t successes = 0;

  for (size_t trial = 0; trial < 4; ++trial)
  {
    RastriginFunction f(2);
    ExponentialSchedule schedule;
    // The convergence is very sensitive to the choices of maxMove and initMove.
    // SA<> sa(schedule, 2000000, 100, 50, 1000, 1e-12, 2, 2.0, 0.5, 0.1);
    SA<> sa(schedule, 2000000, 100, 50, 1000, 1e-12, 2, 2.0, 0.5, 0.1);
    arma::mat coordinates = f.GetInitialPoint();

    const double result = sa.Optimize(f, coordinates);

    if ((std::abs(result) < 1e-3) &&
        (std::abs(coordinates[0]) < 1e-3) &&
        (std::abs(coordinates[1]) < 1e-3))
    {
      ++successes;
      break; // No need to continue.
    }
  }

  BOOST_REQUIRE_GE(successes, 1);
}

BOOST_AUTO_TEST_SUITE_END();
