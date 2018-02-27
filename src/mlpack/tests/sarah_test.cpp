/**
 * @file sarah_test.cpp
 * @author Marcus Edel
 *
 * Test file for SARAH and SARAH+.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/sarah/sarah.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"
#include "test_function_tools.hpp"

using namespace mlpack;
using namespace mlpack::optimization;

BOOST_AUTO_TEST_SUITE(SARAHTest);

/**
 * Run SARAH on logistic regression and make sure the results are
 * acceptable.
 */
BOOST_AUTO_TEST_CASE(SAHRALogisticRegressionTest)
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  // Now run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 45; batchSize += 5)
  {
    SARAH optimizer(0.01, batchSize, 250, 0, 1e-5, true);
    LogisticRegression<> lr(shuffledData, shuffledResponses, optimizer, 0.5);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses);
    BOOST_REQUIRE_CLOSE(acc, 100.0, 1.5); // 1.5% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses);
    BOOST_REQUIRE_CLOSE(testAcc, 100.0, 1.5); // 1.5% error tolerance.
  }
}

/**
 * Run SARAH_Plus on logistic regression and make sure the results are
 * acceptable.
 */
BOOST_AUTO_TEST_CASE(SAHRAPlusLogisticRegressionTest)
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  // Now run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 45; batchSize += 5)
  {
    SARAH_Plus optimizer(0.01, batchSize, 250, 0, 1e-5, true);
    LogisticRegression<> lr(shuffledData, shuffledResponses, optimizer, 0.5);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses);
    BOOST_REQUIRE_CLOSE(acc, 100.0, 1.5); // 1.5% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses);
    BOOST_REQUIRE_CLOSE(testAcc, 100.0, 1.5); // 1.5% error tolerance.
  }
}

BOOST_AUTO_TEST_SUITE_END();
