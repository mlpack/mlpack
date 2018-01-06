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
#include "test_tools.hpp"

#include <mlpack/core/optimizers/sa/sa.hpp>
#include <mlpack/core/optimizers/sa/exponential_schedule.hpp>
#include <mlpack/core/optimizers/lbfgs/test_functions.hpp>
#include <mlpack/core/metrics/ip_metric.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/core/metrics/mahalanobis_distance.hpp>

#include <mlpack/core/math/random.hpp>
#include <mlpack/core/util/log.hpp>

#include <boost/test/unit_test.hpp>

#include <cfloat>

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;
using namespace mlpack::metric;

BOOST_AUTO_TEST_SUITE(SATest);

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
 * The Rastrigrin function, a (not very) simple nonconvex function.  It is
 * defined by
 *
 *   f(x) = 10n + \sum_{i = 1}^{n} (x_i^2 - 10 cos(2 \pi x_i)).
 *
 * It has very many local minima, so finding the true global minimum is
 * difficult.  The function is two-dimensional, and has minimum 0 where
 * x = [0 0].  We are only using it for simulated annealing, so there is no need
 * to implement the gradient.
 */
class RastrigrinFunction
{
 public:
  double Evaluate(const arma::mat& coordinates) const
  {
    double objective = 20; // 10 * n, n = 2.
    objective += std::pow(coordinates[0], 2.0) -
        10 * std::cos(2 * M_PI * coordinates[0]);
    objective += std::pow(coordinates[1], 2.0) -
        10 * std::cos(2 * M_PI * coordinates[1]);

    return objective;
  }

  arma::mat GetInitialPoint() const
  {
    return arma::mat("-3 -3");
  }
};

BOOST_AUTO_TEST_CASE(RastrigrinFunctionTest)
{
  // Simulated annealing isn't guaranteed to converge (except in very specific
  // situations).  If this works 1 of 4 times, I'm fine with that.  All I want
  // to know is that this implementation will escape from local minima.
  size_t successes = 0;

  for (size_t trial = 0; trial < 4; ++trial)
  {
    RastrigrinFunction f;
    ExponentialSchedule schedule;
    // The convergence is very sensitive to the choices of maxMove and initMove.
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
