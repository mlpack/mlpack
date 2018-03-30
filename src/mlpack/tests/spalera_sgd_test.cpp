/**
 * @file spalera_sgd_test.cpp
 * @author Marcus Edel
 *
 * Test file for SGD (stochastic gradient descent).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/spalera_sgd/spalera_sgd.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::distribution;
using namespace mlpack::regression;

BOOST_AUTO_TEST_SUITE(SPALeRASGDTest);

/**
 * Run SPALeRA SGD on logistic regression and make sure the results are
 * acceptable.
 */
BOOST_AUTO_TEST_CASE(LogisticRegressionTest)
{
  // Generate a two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 1.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("9.0 9.0 9.0"), arma::eye<arma::mat>(3, 3));

  arma::mat data(3, 500);
  arma::Row<size_t> responses(500);
  for (size_t i = 0; i < 250; ++i)
  {
    data.col(i) = g1.Random();
    responses[i] = 0;
  }
  for (size_t i = 250; i < 500; ++i)
  {
    data.col(i) = g2.Random();
    responses[i] = 1;
  }

  // Shuffle the dataset.
  arma::uvec indices = arma::shuffle(arma::linspace<arma::uvec>(0,
      data.n_cols - 1, data.n_cols));
  arma::mat shuffledData(3, 500);
  arma::Row<size_t> shuffledResponses(500);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    shuffledData.col(i) = data.col(indices[i]);
    shuffledResponses[i] = responses[indices[i]];
  }

  // Create a test set.
  arma::mat testData(3, 500);
  arma::Row<size_t> testResponses(500);
  for (size_t i = 0; i < 250; ++i)
  {
    testData.col(i) = g1.Random();
    testResponses[i] = 0;
  }
  for (size_t i = 250; i < 500; ++i)
  {
    testData.col(i) = g2.Random();
    testResponses[i] = 1;
  }

  // Now run mini-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 30; batchSize < 50; batchSize += 5)
  {
    SPALeRASGD<> mbsgd(0.05 / batchSize, batchSize, 10000, 1e-4);
    LogisticRegression<> lr(shuffledData, shuffledResponses, mbsgd, 0.5);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses);
    BOOST_REQUIRE_CLOSE(acc, 100.0, 1.5); // 1.5% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses);
    BOOST_REQUIRE_CLOSE(testAcc, 100.0, 2.4); // 2.4% error tolerance.
  }
}

BOOST_AUTO_TEST_SUITE_END();
