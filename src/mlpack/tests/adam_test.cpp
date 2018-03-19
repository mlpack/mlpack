/**
 * @file adam_test.cpp
 * @author Vasanth Kalingeri
 * @author Vivek Pal
 * @author Sourabh Varshney
 * @author Haritha Nair
 *
 * Tests the Adam, AdaMax, AMSGrad, Nadam and NadaMax optimizer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>

#include <mlpack/core/optimizers/adam/adam.hpp>

#include <mlpack/core/optimizers/problems/sgd_test_function.hpp>
#include <mlpack/core/optimizers/problems/colville_function.hpp>
#include <mlpack/core/optimizers/problems/booth_function.hpp>
#include <mlpack/core/optimizers/problems/sphere_function.hpp>
#include <mlpack/core/optimizers/problems/styblinski_tang_function.hpp>
#include <mlpack/core/optimizers/problems/mc_cormick_function.hpp>
#include <mlpack/core/optimizers/problems/matyas_function.hpp>
#include <mlpack/core/optimizers/problems/easom_function.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace arma;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

using namespace mlpack::distribution;
using namespace mlpack::regression;

using namespace mlpack;

BOOST_AUTO_TEST_SUITE(AdamTest);


/**
 * Test the Adam optimizer on the Sphere function.
 */
BOOST_AUTO_TEST_CASE(AdamSphereFunctionTest)
{
  SphereFunction f(2);
  Adam optimizer(0.5, 2, 0.7, 0.999, 1e-8, 500000, 1e-3, false);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  BOOST_REQUIRE_SMALL(coordinates[0], 0.1);
  BOOST_REQUIRE_SMALL(coordinates[1], 0.1);
}

/**
 * Test the Adam optimizer on the Wood function.
 */
BOOST_AUTO_TEST_CASE(AdamStyblinskiTangFunctionTest)
{
  StyblinskiTangFunction f(2);
  Adam optimizer(0.5, 2, 0.7, 0.999, 1e-8, 500000, 1e-3, false);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  BOOST_REQUIRE_CLOSE(coordinates[0], -2.9, 1.0); // 1% error tolerance.
  BOOST_REQUIRE_CLOSE(coordinates[1], -2.9, 1.0); // 1% error tolerance.
}

/**
 * Test the Adam optimizer on the McCormick function.
 */
BOOST_AUTO_TEST_CASE(AdamMcCormickFunctionTest)
{
  McCormickFunction f;
  Adam optimizer(0.5, 1, 0.7, 0.999, 1e-8, 500000, 1e-5, false);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  BOOST_REQUIRE_CLOSE(coordinates[0], -0.547, 3.0); // 3% error tolerance.
  BOOST_REQUIRE_CLOSE(coordinates[1], -1.547, 3.0); // 3% error tolerance.
}

/**
 * Test the Adam optimizer on the Matyas function.
 */
BOOST_AUTO_TEST_CASE(AdamMatyasFunctionTest)
{
  MatyasFunction f;
  Adam optimizer(0.5, 1, 0.7, 0.999, 1e-8, 500000, 1e-5, false);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  // 3% error tolerance.
  BOOST_REQUIRE_CLOSE(std::trunc(100.0 * coordinates[0]) / 100.0, 0.0, 3.0);
  BOOST_REQUIRE_CLOSE(std::trunc(100.0 * coordinates[1]) / 100.0, 0.0, 3.0);
}

/**
 * Test the Adam optimizer on the Easom function.
 */
BOOST_AUTO_TEST_CASE(AdamEasomFunctionTest)
{
  EasomFunction f;
  Adam optimizer(0.2, 1, 0.7, 0.999, 1e-8, 500000, 1e-5, false);

  arma::mat coordinates = arma::mat("2.9; 2.9");
  optimizer.Optimize(f, coordinates);

  // 5% error tolerance.
  BOOST_REQUIRE_CLOSE(std::trunc(100.0 * coordinates[0]) / 100.0, 3.14, 3.0);
  BOOST_REQUIRE_CLOSE(std::trunc(100.0 * coordinates[1]) / 100.0, 3.14, 3.0);
}

/**
 * Test the Adam optimizer on the Booth function.
 */
BOOST_AUTO_TEST_CASE(AdamBoothFunctionTest)
{
  BoothFunction f;
  Adam optimizer(1e-1, 1, 0.7, 0.999, 1e-8, 500000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  BOOST_REQUIRE_CLOSE(coordinates[0], 1.0, 0.2);
  BOOST_REQUIRE_CLOSE(coordinates[1], 3.0, 0.2);
}

/**
 * Tests the Adam optimizer using a simple test function.
 */
BOOST_AUTO_TEST_CASE(SimpleAdamTestFunction)
{
  SGDTestFunction f;
  Adam optimizer(1e-3, 1, 0.9, 0.999, 1e-8, 500000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  BOOST_REQUIRE_SMALL(coordinates[0], 0.1);
  BOOST_REQUIRE_SMALL(coordinates[1], 0.1);
  BOOST_REQUIRE_SMALL(coordinates[2], 0.1);
}

/**
 * Tests the AdaMax optimizer using a simple test function.
 */
BOOST_AUTO_TEST_CASE(SimpleAdaMaxTestFunction)
{
  SGDTestFunction f;
  AdaMax optimizer(2e-3, 1, 0.9, 0.999, 1e-8, 500000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  BOOST_REQUIRE_SMALL(coordinates[0], 0.1);
  BOOST_REQUIRE_SMALL(coordinates[1], 0.1);
  BOOST_REQUIRE_SMALL(coordinates[2], 0.1);
}

/**
 * Tests the AMSGrad optimizer using a simple test function.
 */
BOOST_AUTO_TEST_CASE(SimpleAMSGradTestFunction)
{
  SGDTestFunction f;
  AMSGrad optimizer(1e-3, 1, 0.9, 0.999, 1e-8, 500000, 1e-11, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  BOOST_REQUIRE_SMALL(coordinates[0], 0.1);
  BOOST_REQUIRE_SMALL(coordinates[1], 0.1);
  BOOST_REQUIRE_SMALL(coordinates[2], 0.1);
}

/**
 * Run Adam on logistic regression and make sure the results are acceptable.
 */
BOOST_AUTO_TEST_CASE(AdamLogisticRegressionTest)
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

  Adam adam;
  LogisticRegression<> lr(shuffledData, shuffledResponses, adam, 0.5);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);
  BOOST_REQUIRE_CLOSE(acc, 100.0, 0.3); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses);
  BOOST_REQUIRE_CLOSE(testAcc, 100.0, 0.6); // 0.6% error tolerance.
}

/**
 * Run AdaMax on logistic regression and make sure the results are acceptable.
 */
BOOST_AUTO_TEST_CASE(AdaMaxLogisticRegressionTest)
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

  AdaMax adamax(1e-3, 1, 0.9, 0.999, 1e-8, 5000000, 1e-9, true);
  LogisticRegression<> lr(shuffledData, shuffledResponses, adamax, 0.5);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);
  BOOST_REQUIRE_CLOSE(acc, 100.0, 0.3); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses);
  BOOST_REQUIRE_CLOSE(testAcc, 100.0, 0.6); // 0.6% error tolerance.
}

/**
 * Run AMSGrad on logistic regression and make sure the results are acceptable.
 */
BOOST_AUTO_TEST_CASE(AMSGradLogisticRegressionTest)
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

  AMSGrad amsgrad(1e-3, 1, 0.9, 0.999, 1e-8, 500000, 1e-11, true);
  LogisticRegression<> lr(shuffledData, shuffledResponses, amsgrad, 0.5);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);
  BOOST_REQUIRE_CLOSE(acc, 100.0, 0.3); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses);
  BOOST_REQUIRE_CLOSE(testAcc, 100.0, 0.6); // 0.6% error tolerance.
}

/**
 * Tests the Nadam optimizer using a simple test function.
 */
BOOST_AUTO_TEST_CASE(SimpleNadamTestFunction)
{
  SGDTestFunction f;
  Nadam optimizer(1e-3, 1, 0.9, 0.99, 1e-8, 500000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  BOOST_REQUIRE_SMALL(coordinates[0], 0.1);
  BOOST_REQUIRE_SMALL(coordinates[1], 0.1);
  BOOST_REQUIRE_SMALL(coordinates[2], 0.1);
}

/**
 * Run Nadam on logistic regression and make sure the results are acceptable.
 */
BOOST_AUTO_TEST_CASE(NadamLogisticRegressionTest)
{
  // Generate a two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 1.0 1.0"),
  arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("9.0 9.0 9.0"),
  arma::eye<arma::mat>(3, 3));

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

  Nadam nadam;
  LogisticRegression<> lr(shuffledData, shuffledResponses, nadam, 0.5);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);
  BOOST_REQUIRE_CLOSE(acc, 100.0, 0.3); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses);
  BOOST_REQUIRE_CLOSE(testAcc, 100.0, 0.6); // 0.6% error tolerance.
}

/**
 * Tests the NadaMax optimizer using a simple test function.
 */
BOOST_AUTO_TEST_CASE(SimpleNadaMaxTestFunction)
{
  SGDTestFunction f;
  NadaMax optimizer(1e-3, 1, 0.9, 0.99, 1e-8, 500000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  BOOST_REQUIRE_SMALL(coordinates[0], 0.1);
  BOOST_REQUIRE_SMALL(coordinates[1], 0.1);
  BOOST_REQUIRE_SMALL(coordinates[2], 0.1);
}

/**
 * Run NadaMax on logistic regression and make sure the results are acceptable.
 */
BOOST_AUTO_TEST_CASE(NadaMaxLogisticRegressionTest)
{
  // Generate a two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 1.0 1.0"),
  arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("9.0 9.0 9.0"),
  arma::eye<arma::mat>(3, 3));

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

  NadaMax nadamax;
  LogisticRegression<> lr(shuffledData, shuffledResponses, nadamax, 0.5);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);
  BOOST_REQUIRE_CLOSE(acc, 100.0, 0.3); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses);
  BOOST_REQUIRE_CLOSE(testAcc, 100.0, 0.6); // 0.6% error tolerance.
}

/**
 * Tests the OptimisticAdam optimizer using a simple test function.
 */
BOOST_AUTO_TEST_CASE(SimpleOptimisticAdamTestFunction)
{
  SGDTestFunction f;
  OptimisticAdam optimizer(1e-2, 1, 0.9, 0.99, 1e-8);

  arma::mat coordinates = f.GetInitialPoint();
  optimizer.Optimize(f, coordinates);

  BOOST_REQUIRE_SMALL(coordinates[0], 0.1);
  BOOST_REQUIRE_SMALL(coordinates[1], 0.1);
  BOOST_REQUIRE_SMALL(coordinates[2], 0.1);
}

/**
 * Run OptimisticAdam on logistic regression and make sure the results are acceptable.
 */
BOOST_AUTO_TEST_CASE(OptimisticAdamLogisticRegressionTest)
{
  // Generate a two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 1.0 1.0"),
  arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("9.0 9.0 9.0"),
  arma::eye<arma::mat>(3, 3));

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

  OptimisticAdam optimisticAdam;
  LogisticRegression<> lr(shuffledData, shuffledResponses, optimisticAdam, 0.5);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);
  BOOST_REQUIRE_CLOSE(acc, 100.0, 0.3); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses);
  BOOST_REQUIRE_CLOSE(testAcc, 100.0, 0.6); // 0.6% error tolerance.
}

BOOST_AUTO_TEST_SUITE_END();
