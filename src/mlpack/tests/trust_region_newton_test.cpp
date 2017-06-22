/**
 * @file trust_region_newton_test.cpp
 * @author Marcus Edel
 *
 * Tests the Adam and AdaMax optimizer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/core/optimizers/trust_region_newton/trust_region_newton.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace arma;
using namespace mlpack::optimization;

using namespace mlpack::distribution;
using namespace mlpack::regression;

using namespace mlpack;

BOOST_AUTO_TEST_SUITE(TrustRegionNewtonTest);

// Test training of logistic regression on two Gaussians and ensure it's
// properly separable using the Trust Region Newton method.
BOOST_AUTO_TEST_CASE(LogisticRegressionTrustRegionNewtonGaussianTest)
{
  // Generate a two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 1.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("9.0 9.0 9.0"), arma::eye<arma::mat>(3, 3));

  arma::mat data(3, 1000);
  arma::Row<size_t> responses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    data.col(i) = g1.Random();
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    data.col(i) = g2.Random();
    responses[i] = 1;
  }

  // Now train a logistic regression object on it.
  LogisticRegression<> lr(data.n_rows, 0.5);
  lr.Train<TrustRegionNewton>(data, responses);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);

  BOOST_REQUIRE_CLOSE(acc, 100.0, 0.3); // 0.3% error tolerance.

  // Create a test set.
  for (size_t i = 0; i < 500; ++i)
  {
    data.col(i) = g1.Random();
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    data.col(i) = g2.Random();
    responses[i] = 1;
  }

  // Ensure that the error is close to zero.
  const double testAcc = lr.ComputeAccuracy(data, responses);

  BOOST_REQUIRE_CLOSE(testAcc, 100.0, 0.6); // 0.6% error tolerance.
}

BOOST_AUTO_TEST_SUITE_END();
