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
#include <mlpack/core/optimizers/problems/sphere_function.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;
using namespace mlpack::distribution;
using namespace mlpack::regression;

BOOST_AUTO_TEST_SUITE(SPSATest);

/**
 * Tests the SPSA optimizer using the Sphere Function.
 */
BOOST_AUTO_TEST_CASE(SPSASphereFunctionTest)
{
  for (size_t i = 10; i <= 50; i++)
  {
    SphereFunction h(i);

    SPSA optimiser(0.1, 0.102, 0.16,
                   0.3, 100000);

    arma::mat coordinates = h.GetInitialPoint();
    double result = optimiser.Optimize(h, coordinates);

    if (i <= 30)
      BOOST_REQUIRE_CLOSE(result, 0.0, 10000);
    else if (i <= 34 || i == 36 || i == 39)
      BOOST_REQUIRE_CLOSE(result, 8e-33, 10000);
    else
      BOOST_REQUIRE_CLOSE(result, 3e-32, 100000);

    BOOST_REQUIRE_SMALL(coordinates[0], 1e-3);
    BOOST_REQUIRE_SMALL(coordinates[1], 1e-7);
    BOOST_REQUIRE_SMALL(coordinates[2], 1e-7);
  }
}

BOOST_AUTO_TEST_CASE(SPSALogisticRegressionTest)
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

  // Shuffle the dataset.
  arma::uvec indices = arma::shuffle(arma::linspace<arma::uvec>(0,
      data.n_cols - 1, data.n_cols));
  arma::mat shuffledData(3, 1000);
  arma::Row<size_t> shuffledResponses(1000);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    shuffledData.col(i) = data.col(indices[i]);
    shuffledResponses[i] = responses[indices[i]];
  }

  // Create a test set.
  arma::mat testData(3, 1000);
  arma::Row<size_t> testResponses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    testData.col(i) = g1.Random();
    testResponses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    testData.col(i) = g2.Random();
    testResponses[i] = 1;
  }

  SPSA optimiser(0.1, 0.102, 0.16,
               0.3, 100000);

  LogisticRegression<> lr(shuffledData, shuffledResponses, optimiser, 0.5);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);
  BOOST_REQUIRE_CLOSE(acc, 100.0, 0.3); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses);
  BOOST_REQUIRE_CLOSE(testAcc, 100.0, 0.6); // 0.6% error tolerance.
}

BOOST_AUTO_TEST_SUITE_END();
