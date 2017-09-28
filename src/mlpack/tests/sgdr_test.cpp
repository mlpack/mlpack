/**
 * @file sgdr_test.cpp
 * @author Marcus Edel
 *
 * Test file for SGDR.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/sgdr/cyclical_decay.hpp>
#include <mlpack/core/optimizers/sgdr/sgdr.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;

using namespace mlpack::distribution;
using namespace mlpack::regression;

BOOST_AUTO_TEST_SUITE(SGDRTest);

/*
 * Test that the step size resets after a specified number of epochs.
 */
BOOST_AUTO_TEST_CASE(CyclicalResetTest)
{
  const double stepSize = 0.5;
  arma::mat iterate;

  // Now run cyclical decay policy with a couple of multiplicators and initial
  // restarts.
  for (size_t restart = 5; restart < 100; restart += 10)
  {
    for (size_t mult = 2; mult < 5; ++mult)
    {
      double epochStepSize = stepSize;

      CyclicalDecay cyclicalDecay(restart, double(mult), stepSize);
      cyclicalDecay.EpochBatches() = (double) 1000 / 10;

      // Create all restart epochs.
      arma::Col<size_t> nextRestart(1000 / 10 /  mult);
      nextRestart(0) = restart;
      for (size_t j = 1; j < nextRestart.n_elem; ++j)
        nextRestart(j) = nextRestart(j - 1) * mult;

      for (size_t i = 0; i < 1000; ++i)
      {
        cyclicalDecay.Update(iterate, epochStepSize, iterate);
        if (i <= restart || arma::accu(arma::find(nextRestart == i)) > 0)
        {
          BOOST_CHECK_EQUAL(epochStepSize, stepSize);
        }
      }
    }
  }
}

/**
 * Run SGDR on logistic regression and make sure the results are acceptable.
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

  // Now run SGDR with a couple of batch sizes.
  for (size_t batchSize = 5; batchSize < 50; batchSize += 5)
  {
    SGDR<> sgdr(50, 2.0, batchSize, 0.01, 10000, 1e-3);
    LogisticRegression<> lr(shuffledData, shuffledResponses, sgdr, 0.5);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses);
    BOOST_REQUIRE_CLOSE(acc, 100.0, 0.3); // 0.3% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses);
    BOOST_REQUIRE_CLOSE(testAcc, 100.0, 0.6); // 0.6% error tolerance.
  }
}

BOOST_AUTO_TEST_SUITE_END();
