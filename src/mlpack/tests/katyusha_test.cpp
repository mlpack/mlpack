/**
 * @file katyusha_test.cpp
 * @author Marcus Edel
 *
 * Test file for Katyusha.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/katyusha/katyusha.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"
#include "test_function_tools.hpp"

using namespace mlpack;
using namespace mlpack::optimization;

BOOST_AUTO_TEST_SUITE(KatyushaTest);

/**
 * Run Katyusha on logistic regression and make sure the results are acceptable.
 */
BOOST_AUTO_TEST_CASE(KatyushaLogisticRegressionTest)
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  // Now run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 30; batchSize < 45; batchSize += 5)
  {
    Katyusha optimizer(1.0, 10.0, batchSize, 100, 0, 1e-10, true);
    LogisticRegression<> lr(shuffledData, shuffledResponses, optimizer, 0.5);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses);
    BOOST_REQUIRE_CLOSE(acc, 100.0, 1.5); // 1.5% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses);
    BOOST_REQUIRE_CLOSE(testAcc, 100.0, 1.5); // 1.5% error tolerance.
  }
}

/**
 * Run Proximal Katyusha on logistic regression and make sure the results are
 * acceptable.
 */
BOOST_AUTO_TEST_CASE(KatyushaProximalLogisticRegressionTest)
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  // Now run big-batch SGD with a couple of batch sizes.
  for (size_t batchSize = 30; batchSize < 45; batchSize += 5)
  {
    KatyushaProximal optimizer(1.0, 10.0, batchSize, 100, 0, 1e-10, true);
    LogisticRegression<> lr(shuffledData, shuffledResponses, optimizer, 0.5);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses);
    BOOST_REQUIRE_CLOSE(acc, 100.0, 1.5); // 1.5% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses);
    BOOST_REQUIRE_CLOSE(testAcc, 100.0, 1.5); // 1.5% error tolerance.
  }
}

BOOST_AUTO_TEST_SUITE_END();
