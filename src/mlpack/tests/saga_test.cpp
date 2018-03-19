/**
 * @file saga_test.cpp
 * @author Prabhat Sharma
 *
 * Test file for SAGA.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core/optimizers/saga/saga.hpp>
#include <mlpack/core/optimizers/problems/sgd_test_function.hpp>
#include <mlpack/core/optimizers/problems/generalized_rosenbrock_function.hpp>

#include <boost/test/unit_test.hpp>
#include "test_function_tools.hpp"

using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

BOOST_AUTO_TEST_SUITE(SAGATest);

/**
 * Run SAGA on logistic regression and make sure the results are acceptable.
 */
BOOST_AUTO_TEST_CASE(SAGALogisticRegressionTest)
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
      responses, testResponses, shuffledResponses);

  // Now run mini-batch SAGA with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 50; batchSize += 5)
  {
    SAGA optimizer(0.001, batchSize, 250, 1e-3, false);
    LogisticRegression<> lr(shuffledData, shuffledResponses, optimizer, 0.5);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses);
    BOOST_REQUIRE_CLOSE(acc, 100.0, 1.5); // 1.5% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses);
    BOOST_REQUIRE_CLOSE(testAcc, 100.0, 1.5); // 1.5% error tolerance.
  }
}

/**
 * Run SAGA on Generalized Rosenbrock Function and make sure the
 * results are acceptable.
 */
BOOST_AUTO_TEST_CASE(GeneralizedRosenbrockTest)
{
  // Loop over several variants.
  for (size_t i = 5; i < 25; i += 2)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction f(i);

    SAGA optimizer(0.0001, 1, 0, 1e-15, false);

    arma::mat coordinates = f.GetInitialPoint();
    double result = optimizer.Optimize(f, coordinates);

    BOOST_REQUIRE_SMALL(result, 1e-10);
    for (size_t j = 0; j < i; ++j)
      BOOST_REQUIRE_CLOSE(coordinates[j], (double) 1.0, 1e-3);
  }
}

BOOST_AUTO_TEST_SUITE_END();
