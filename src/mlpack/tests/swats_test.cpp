/**
 * @file swats_test.cpp
 * @author Marcus Edel
 *
 * Test file for the SWATS optimizer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/swats/swats.hpp>

#include <mlpack/core/optimizers/problems/sphere_function.hpp>
#include <mlpack/core/optimizers/problems/styblinski_tang_function.hpp>
#include <mlpack/core/optimizers/problems/sgd_test_function.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace arma;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

using namespace mlpack::distribution;
using namespace mlpack::regression;

using namespace mlpack;

BOOST_AUTO_TEST_SUITE(SWATSTest);

/**
 * Test the SWATS optimizer on the simple SGD function.
 */
BOOST_AUTO_TEST_CASE(SWATSSGDFunction)
{
  SGDTestFunction f;
  SWATS optimizer(1e-3, 1, 0.9, 0.999, 1e-6, 400000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  BOOST_REQUIRE_SMALL(coordinates[0], 0.1);
  BOOST_REQUIRE_SMALL(coordinates[1], 0.1);
  BOOST_REQUIRE_SMALL(coordinates[2], 0.1);
}

/**
 * Run SWATS on logistic regression and make sure the results are acceptable.
 */
BOOST_AUTO_TEST_CASE(SWATSLogisticRegressionTest)
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

  SWATS optimizer(1e-3, 1, 0.9, 0.999, 1e-6, 500000, 1e-9, true);
  LogisticRegression<> lr(shuffledData, shuffledResponses, optimizer, 0.5);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);
  BOOST_REQUIRE_CLOSE(acc, 100.0, 0.3); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses);
  BOOST_REQUIRE_CLOSE(testAcc, 100.0, 0.6); // 0.6% error tolerance.
}

/**
 * Test the SWATS optimizer on the Sphere function.
 */
BOOST_AUTO_TEST_CASE(SWATSSphereFunctionTest)
{
  SphereFunction f(2);
  SWATS optimizer(1e-3, 2, 0.9, 0.999, 1e-6, 500000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  BOOST_REQUIRE_SMALL(coordinates[0], 0.1);
  BOOST_REQUIRE_SMALL(coordinates[1], 0.1);
}

/**
 * Test the SWATS optimizer on the Wood function.
 */
BOOST_AUTO_TEST_CASE(SWATSStyblinskiTangFunctionTest)
{
  StyblinskiTangFunction f(2);
  SWATS optimizer(1e-3, 2, 0.9, 0.999, 1e-6, 500000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  BOOST_REQUIRE_CLOSE(coordinates[0], -2.9, 1.0); // 1% error tolerance.
  BOOST_REQUIRE_CLOSE(coordinates[1], -2.9, 1.0); // 1% error tolerance.
}

BOOST_AUTO_TEST_SUITE_END();
