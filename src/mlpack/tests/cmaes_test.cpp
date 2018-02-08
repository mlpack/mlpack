/**
 * @file cmaes_test.cpp
 * @author Marcus Edel
 * @author Kartik Nighania
 *
 * Test file for CMA-ES.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/cmaes/cmaes.hpp>
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

BOOST_AUTO_TEST_SUITE(CMAESTest);

/**
 * Tests the CMA-ES optimizer using a simple test function.
 */
BOOST_AUTO_TEST_CASE(SimpleTestFunction)
{
  SGDTestFunction f;
  CMAES<> optimizer(0, -1, 1, 32, 200, -1);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  BOOST_REQUIRE_SMALL(coordinates[0], 0.003);
  BOOST_REQUIRE_SMALL(coordinates[1], 0.003);
  BOOST_REQUIRE_SMALL(coordinates[2], 0.003);
}

/**
 * Create the data for the logistic regression test case.
 */
void CreateLogisticRegressionTestData(arma::mat& data,
                                      arma::mat& testData,
                                      arma::mat& shuffledData,
                                      arma::Row<size_t>& responses,
                                      arma::Row<size_t>& testResponses,
                                      arma::Row<size_t>& shuffledResponses)
{
  // Generate a two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 1.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("9.0 9.0 9.0"), arma::eye<arma::mat>(3, 3));

  data = arma::mat(3, 1000);
  responses = arma::Row<size_t>(1000);
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
  shuffledData = arma::mat(3, 1000);
  shuffledResponses = arma::Row<size_t>(1000);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    shuffledData.col(i) = data.col(indices[i]);
    shuffledResponses[i] = responses[indices[i]];
  }

  // Create a test set.
  testData = arma::mat(3, 1000);
  testResponses = arma::Row<size_t>(1000);
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
}

/**
 * Run CMA-ES with the full selection policy on logistic regression and
 * make sure the results are acceptable.
 */
BOOST_AUTO_TEST_CASE(CMAESLogisticRegressionTest)
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  CreateLogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  CMAES<> cmaes(0, -1, 1, 32, 200, 1e-3);
  LogisticRegression<> lr(shuffledData, shuffledResponses, cmaes, 0.5);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);
  BOOST_REQUIRE_CLOSE(acc, 100.0, 0.3); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses);
  BOOST_REQUIRE_CLOSE(testAcc, 100.0, 0.6); // 0.6% error tolerance.
}

/**
 * Run CMA-ES with the random selection policy on logistic regression and
 * make sure the results are acceptable.
 */
BOOST_AUTO_TEST_CASE(ApproxCMAESLogisticRegressionTest)
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  CreateLogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  ApproxCMAES<> cmaes(0, -1, 1, 32, 200, 1e-3);
  LogisticRegression<> lr(shuffledData, shuffledResponses, cmaes, 0.5);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);
  BOOST_REQUIRE_CLOSE(acc, 100.0, 0.3); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses);
  BOOST_REQUIRE_CLOSE(testAcc, 100.0, 0.6); // 0.6% error tolerance.
}

BOOST_AUTO_TEST_SUITE_END();
