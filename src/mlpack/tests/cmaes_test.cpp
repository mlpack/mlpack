/**
 * @file cmaes_test.cpp
 * @author Kartik Nighania Mentor Marcus Edel
 *
 * Test file for CMAES (Covariance Matrix Adaptation Evolution Strategy).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/core/optimizers/cmaes/cmaes.hpp>
#include <mlpack/core/optimizers/lbfgs/test_functions.hpp>
#include <mlpack/core/optimizers/sgd/test_function.hpp>

#include <mlpack/methods/logistic_regression/logistic_regression.hpp>

#include <mlpack/core/optimizers/sgd/update_policies/vanilla_update.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

using namespace mlpack::distribution;
using namespace mlpack::regression;

BOOST_AUTO_TEST_SUITE(CMAESTest);

BOOST_AUTO_TEST_CASE(SimpleCMAESTestFunction)
{
  SGDTestFunction test;

  CMAES s(3, 0.5, 0.3, 10000, 1e-13, 1e-13);

  arma::mat coordinates(3, 1);
  double result = s.Optimize(test, coordinates);

  BOOST_REQUIRE_CLOSE(result, -1.0, 0.05);
  BOOST_REQUIRE_SMALL(coordinates[0], 1e-3);
  BOOST_REQUIRE_SMALL(coordinates[1], 1e-3);
  BOOST_REQUIRE_SMALL(coordinates[2], 1e-3);
}

BOOST_AUTO_TEST_CASE(LogisticRegressionTestWithCMAES)
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

  int dim = shuffledData.n_rows + 1;
  CMAES test1(dim, 0.5, 0.3, 10000, 1e-10, 1e-10);

  LogisticRegression<> lr(shuffledData, shuffledResponses, test1, 0.5);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);
  BOOST_REQUIRE_CLOSE(acc, 100.0, 0.3); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses);
  BOOST_REQUIRE_CLOSE(testAcc, 100.0, 0.6); // 0.6% error tolerance.
}

BOOST_AUTO_TEST_CASE(rosenbrockFunctionCMAES)
{
  mlpack::math::RandomSeed(std::time(NULL));

  // Loop over several variants.
  for (size_t i = 5; i < 30; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction f(i);

    CMAES s(i, 0.5, 0.3, 100000, 1e-16, 1e-16);

    arma::mat coordinates = f.GetInitialPoint();
    double result = s.Optimize(f, coordinates);

    BOOST_REQUIRE_SMALL(result, 1e-5);
    for (size_t j = 0; j < i; ++j)
      BOOST_REQUIRE_CLOSE(coordinates[j], (double) 1.0, 1e-3);
  }
}

BOOST_AUTO_TEST_SUITE_END();
