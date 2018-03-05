/**
 * @file pso_test.cpp
 * @author Adeel Ahmad
 *
 * Test file for PSO.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/core/optimizers/pso/pso.hpp>
#include <mlpack/core/optimizers/pso/test_function.hpp>

#include <mlpack/core/optimizers/problems/wood_function.cpp>
#include <mlpack/core/optimizers/problems/booth_function.cpp>
#include <mlpack/core/optimizers/problems/easom_function.cpp>
#include <mlpack/core/optimizers/problems/bukin_function.cpp>
#include <mlpack/core/optimizers/problems/matyas_function.cpp>
#include <mlpack/core/optimizers/problems/colville_function.cpp>
#include <mlpack/core/optimizers/problems/sgd_test_function.hpp>
#include <mlpack/core/optimizers/problems/eggholder_function.cpp>
#include <mlpack/core/optimizers/problems/rosenbrock_function.cpp>
#include <mlpack/core/optimizers/problems/mc_cormick_function.cpp>

#include <mlpack/methods/logistic_regression/logistic_regression.hpp>

#include <boost/test/unit_test.hpp>

using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::distribution;
using namespace mlpack::regression;

BOOST_AUTO_TEST_SUITE(PSOTest);

/**
 * Simple test case for PSO.
 */
BOOST_AUTO_TEST_CASE(SimpleTest)
{
  PSO optimizer;
  PSOTestFunction f;

  arma::mat iterate;
  iterate << 0.5828 << 0.0654 << 0.8817;

  double result = optimizer.Optimize(f, iterate);

  BOOST_REQUIRE_SMALL(result, 1e-5);
}

/**
 * Test for minimizing the Rosenbrock function.
 */
BOOST_AUTO_TEST_CASE(RosenbrockFunctionTest)
{
  PSO optimizer;
  RosenbrockFunction f;

  arma::mat iterate;
  iterate << 0.5828 << 0.0654 << 0.8817 << 0.3461 << arma::endr
          << 0.2351 << 0.0034 << 0.1641 << 0.7621;

  double result = optimizer.Optimize(f, iterate);

  BOOST_REQUIRE_SMALL(result, 1e-5);
}

/**
 * Test for minimizing Matyas function.
 */
BOOST_AUTO_TEST_CASE(MatyasFunctionTest)
{
  PSO optimizer;
  MatyasFunction f;

  arma::mat iterate;
  iterate << 0.5828 << 0.0654 << 0.8817 << 0.3461 << arma::endr
          << 0.2351 << 0.0034 << 0.1641 << 0.7621;

  double result = optimizer.Optimize(f, iterate);

  BOOST_REQUIRE_SMALL(result, 1e-5);
}

/**
 * Test for minimizing the Booth function.
 */
BOOST_AUTO_TEST_CASE(BoothFunctionTest)
{
  PSO optimizer(60, 0.9, 0.5, 0.3, 200, 1e-3);
  BoothFunction f;

  arma::mat iterate;
  iterate << 1 << 3;

  double result = optimizer.Optimize(f, iterate);

  BOOST_REQUIRE_CLOSE(result, 0, 1e-3);
}

/**
 * Test for the McCormick function.
 */
BOOST_AUTO_TEST_CASE(McCormickTest)
{
  PSO optimizer;
  McCormickFunction f;

  arma::mat iterate;
  iterate << -0.54719 << -1.54719;

  double result = optimizer.Optimize(f, iterate);

  BOOST_REQUIRE_CLOSE(result, -1.9133, 1e-2);
}

/**
 * Test for the Eggholder function.
 */
BOOST_AUTO_TEST_CASE(EggholderFunctionTest)
{
  PSO optimizer;
  EggholderFunction f;

  arma::mat iterate;
  iterate << 512 << 404.2319;

  double result = optimizer.Optimize(f, iterate);

  BOOST_REQUIRE_CLOSE(result, -959.6407, 1e-5);
}

/**
 * Test for the Easom function.
 */
BOOST_AUTO_TEST_CASE(EasomFunctionTest)
{
  PSO optimizer;
  EasomFunction f;

  arma::mat iterate;
  iterate << 3.14 << 3.14;

  double result = optimizer.Optimize(f, iterate);

  BOOST_REQUIRE_CLOSE(result, -1, 1e-3);
}

/**
 * Test for the Colville function.
 */
BOOST_AUTO_TEST_CASE(ColvilleFunctionTest)
{
  PSO optimizer;
  ColvilleFunction f;

  arma::mat iterate;
  iterate << 1 << 1 << 1 << 1;

  double result = optimizer.Optimize(f, iterate);

  BOOST_REQUIRE_CLOSE(result, 0, 1e-4);
}

/**
 * Test for the Bukin function.
 */
BOOST_AUTO_TEST_CASE(BukinFunctionTest)
{
  PSO optimizer;
  BukinFunction f;

  arma::mat iterate;
  iterate << -10 << 1;

  double result = optimizer.Optimize(f, iterate);

  BOOST_REQUIRE_CLOSE(result, 0, 1e-4);
}

/**
 * Test for logistic regression.
 */
BOOST_AUTO_TEST_CASE(LogisticRegressionTest)
{
  // Set the random seed.
  math::RandomSeed(std::time(NULL));

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

  ConstrictionPSO optimizer(30, 0.9, 2.05, 2.05, 10000, 1e-3);
  LogisticRegression<> lr(shuffledData, shuffledResponses, optimizer, 0.5);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);
  BOOST_REQUIRE_CLOSE(acc, 100.0, 0.3); // 0.3% error tolerance.

  const double testAcc = lr.ComputeAccuracy(testData, testResponses);
  BOOST_REQUIRE_CLOSE(testAcc, 100.0, 0.6); // 0.6% error tolerance.
}

BOOST_AUTO_TEST_SUITE_END();
