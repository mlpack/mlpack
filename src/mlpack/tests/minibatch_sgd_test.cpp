/**
 * @file minibatch_sgd_test.cpp
 * @author Ryan Curtin
 *
 * Test file for minibatch SGD.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/core/optimizers/minibatch_sgd/minibatch_sgd.hpp>
#include <mlpack/core/optimizers/lbfgs/test_functions.hpp>
#include <mlpack/core/optimizers/sgd/test_function.hpp>

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

BOOST_AUTO_TEST_SUITE(MiniBatchSGDTest);

/**
 * If the batch size is 1, and we aren't shuffling, we should get the exact same
 * results as regular SGD.
 */
BOOST_AUTO_TEST_CASE(SGDSimilarityTest)
{
  SGDTestFunction f;
  SGD<SGDTestFunction> s(f, 0.0003, 5000000, 1e-9, false);
  MiniBatchSGD<SGDTestFunction> ms(f, 1, 0.0003, 5000000, 1e-9, false);

  arma::mat sCoord = f.GetInitialPoint();
  arma::mat msCoord = f.GetInitialPoint();

  const double sResult = s.Optimize(sCoord);
  const double msResult = s.Optimize(msCoord);

  BOOST_REQUIRE_CLOSE(sResult, msResult, 1e-8);
  BOOST_REQUIRE_CLOSE(sCoord[0], msCoord[0], 1e-8);
  BOOST_REQUIRE_CLOSE(sCoord[1], msCoord[1], 1e-8);
  BOOST_REQUIRE_CLOSE(sCoord[2], msCoord[2], 1e-8);
}

/*
BOOST_AUTO_TEST_CASE(SimpleSGDTestFunction)
{
  SGDTestFunction f;
  // Batch size of 3.
  MiniBatchSGD<SGDTestFunction> s(f, 3, 0.0005, 2000000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  double result = s.Optimize(coordinates);

  BOOST_REQUIRE_CLOSE(result, -1.0, 0.05);
  BOOST_REQUIRE_SMALL(coordinates[0], 1e-3);
  BOOST_REQUIRE_SMALL(coordinates[1], 1e-7);
  BOOST_REQUIRE_SMALL(coordinates[2], 1e-7);
}
*/

/**
 * Run mini-batch SGD on logistic regression and make sure the results are
 * acceptable.
 */
BOOST_AUTO_TEST_CASE(LogisticRegressionTest)
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

  // Now run mini-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 5; batchSize < 50; batchSize += 5)
  {
    LogisticRegression<> lr(shuffledData.n_rows, 0.5);

    LogisticRegressionFunction<> lrf(shuffledData, shuffledResponses, 0.5);
    MiniBatchSGD<LogisticRegressionFunction<>> mbsgd(lrf, batchSize);
    lr.Train(mbsgd);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses);
    BOOST_REQUIRE_CLOSE(acc, 100.0, 0.3); // 0.3% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses);
    BOOST_REQUIRE_CLOSE(testAcc, 100.0, 0.6); // 0.6% error tolerance.
  }
}

/**
 * Run mini-batch SGD on a simple test function and make sure the last batch
 * size is handled correctly.
 *
 * When using a batchsize that fulfilled the constraint:
 * (numFunctions % batchSize) == 1 we have to make sure that the last batch size
 * isn't zero.
 */
BOOST_AUTO_TEST_CASE(ZeroBatchSizeTest)
{
  // Create the generalized Rosenbrock function.
  GeneralizedRosenbrockFunction f(10);

  MiniBatchSGD<GeneralizedRosenbrockFunction> s(
      f, f.NumFunctions() - 1, 0.01, 3);

  arma::mat coordinates = f.GetInitialPoint();
  s.Optimize(coordinates);

  const bool finite = coordinates.is_finite();
  BOOST_REQUIRE_EQUAL(finite, true);
}

BOOST_AUTO_TEST_SUITE_END();
