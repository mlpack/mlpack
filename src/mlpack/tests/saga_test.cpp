/**
 * @file saga_test.cpp
 * @authors Prabhat Sharma and Marcus Edel
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
#include <mlpack/core/optimizers/problems/sphere_function.hpp>

#include <boost/test/unit_test.hpp>
#include "test_function_tools.hpp"

using namespace mlpack;
using namespace arma;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

BOOST_AUTO_TEST_SUITE(SAGATypeTest);

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
* Run SAGA on Sphere function and make sure the results are acceptable.
*/
BOOST_AUTO_TEST_CASE(SphereFunctionSAGATest)
{
  // Loop over several variants.
  for (size_t i = 50; i < 100; i += 10)
  {
    SphereFunction f(i);

    double stepSize = 0.00001 * i;

    SAGA optimizer(stepSize, 5, 500000, 1e-7, false);

    arma::mat coordinates = f.GetInitialPoint();
    double result = optimizer.Optimize(f, coordinates);

    BOOST_REQUIRE_SMALL(result, 0.1);
    for (size_t j = 0; j < i; ++j)
      BOOST_REQUIRE_SMALL(coordinates[j], (double) 0.001);
  }
}
/**
* Run SAG on logistic regression and make sure the results are acceptable.
*/
BOOST_AUTO_TEST_CASE(SAGLogisticRegressionTest)
{
  arma::mat data, testData, shuffledData;
  arma::Row<size_t> responses, testResponses, shuffledResponses;

  LogisticRegressionTestData(data, testData, shuffledData,
                             responses, testResponses, shuffledResponses);

  // Now run mini-batch SAGA with a couple of batch sizes.
  for (size_t batchSize = 35; batchSize < 50; batchSize += 5)
  {
    SAG optimizer(0.001, batchSize, 250, 1e-3, false);
    LogisticRegression<> lr(shuffledData, shuffledResponses, optimizer, 0.5);

    // Ensure that the error is close to zero.
    const double acc = lr.ComputeAccuracy(data, responses);
    BOOST_REQUIRE_CLOSE(acc, 100.0, 1.5); // 1.5% error tolerance.

    const double testAcc = lr.ComputeAccuracy(testData, testResponses);
    BOOST_REQUIRE_CLOSE(testAcc, 100.0, 1.5); // 1.5% error tolerance.
  }
}

/**
* Run SAG on Sphere function and make sure the results are acceptable.
*/
BOOST_AUTO_TEST_CASE(SphereFunctionSAGTest)
{
  // Loop over several variants.
  for (size_t i = 50; i < 100; i += 10)
  {
    SphereFunction f(i);

    double stepSize = 0.00001 * i;

    SAG optimizer(stepSize, 5, 500000, 1e-7, false);

    arma::mat coordinates = f.GetInitialPoint();
    double result = optimizer.Optimize(f, coordinates);

    BOOST_REQUIRE_SMALL(result, 0.1);
    for (size_t j = 0; j < i; ++j)
      BOOST_REQUIRE_SMALL(coordinates[j], (double) 0.001);
  }
}
BOOST_AUTO_TEST_SUITE_END();
