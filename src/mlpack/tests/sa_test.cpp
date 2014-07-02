/*
 * @file sa_test.cpp
 * @auther Zhihao Lou
 *
 * Test file for SA (simulated annealing).
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/sa/sa.hpp>
#include <mlpack/core/optimizers/sa/exponential_schedule.hpp>
#include <mlpack/core/optimizers/sa/laplace_distribution.hpp>
#include <mlpack/core/optimizers/lbfgs/test_functions.hpp>

#include <mlpack/core/metrics/ip_metric.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/core/metrics/mahalanobis_distance.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;
using namespace mlpack::metric;

BOOST_AUTO_TEST_SUITE(SATest);

BOOST_AUTO_TEST_CASE(GeneralizedRosenbrockTest)
{
  size_t dim = 50;
  GeneralizedRosenbrockFunction f(dim);

  LaplaceDistribution moveDist;
  ExponentialSchedule schedule(1e-5);
  SA<GeneralizedRosenbrockFunction, LaplaceDistribution, ExponentialSchedule> 
      sa(f, moveDist, schedule, 1000.,1000, 100, 1e-9, 3, 20, 0.3, 0.3, 10000000);
  arma::mat coordinates = f.GetInitialPoint();
  double result = sa.Optimize(coordinates);

  BOOST_REQUIRE_SMALL(result, 1e-6);
  for (size_t j = 0; j < dim; ++j)
      BOOST_REQUIRE_CLOSE(coordinates[j], (double) 1.0, 1e-2);
}

BOOST_AUTO_TEST_SUITE_END();
