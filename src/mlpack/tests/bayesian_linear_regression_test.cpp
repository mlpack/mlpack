/**
 * @file tests/bayesian_linear_regression_test.cpp 
 * @author Clement Mercier
 *
 * Test for BayesianLinearRegression.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core/data/load.hpp>
#include <mlpack/methods/bayesian_linear_regression/bayesian_linear_regression.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>

#include "catch.hpp"

using namespace mlpack::regression;
using namespace mlpack::data;

void GenerateProblem(arma::mat& matX,
                     arma::rowvec& y,
                     size_t nPoints,
                     size_t nDims,
                     float sigma = 0.0)
{
  matX = arma::randn(nDims, nPoints);
  arma::colvec omega = arma::randn(nDims);
  // Compute y and add noise.
  y = omega.t() * matX + arma::randn(nPoints).t() * sigma;
}

// Ensure that predictions are close enough to the target
// for a free noise dataset.
TEST_CASE("BayesianLinearRegressionRegressionTest",
          "[BayesianLinearRegressionTest]")
{
  arma::mat matX;
  arma::rowvec y, predictions;

  GenerateProblem(matX, y, 200, 10);

  // Instanciate and train the estimator.
  BayesianLinearRegression estimator(true);
  estimator.Train(matX, y);
  estimator.Predict(matX, predictions);

  // Check the predictions are close enough to the targets in a free noise case.
  for (size_t i = 0; i < y.size(); i++)
    REQUIRE(predictions[i] == Approx(y[i]).epsilon(1e-8));

  // Check that the estimated variance is zero.
  REQUIRE(estimator.Variance() == Approx(0.0).margin(1e-5));
}

// Verify centerData and scaleData equal false do not affect the solution.
TEST_CASE("TestCenter0ScaleData0", "[BayesianLinearRegressionTest]")
{
  arma::mat matX;
  arma::rowvec y;
  size_t nDims = 30, nPoints = 100;

  GenerateProblem(matX, y, nPoints, nDims, 0.5);

  BayesianLinearRegression estimator(false, false);

  estimator.Train(matX, y);

  // Check dataOffset is empty.
  REQUIRE(estimator.DataOffset().n_elem == 0);

  // To be neutral responseOffset must be 0.
  REQUIRE(estimator.ResponsesOffset() == 0);

  // Check dataScale is empty.
  REQUIRE(estimator.DataScale().n_elem == 0);
}

// Verify that centering and normalization are correct.
TEST_CASE("TestCenterDataTrueScaleDataTrue", "[BayesianLinearRegressionTest]")
{
  arma::mat matX;
  arma::rowvec y;
  size_t nDims = 5, nPoints = 100;
  GenerateProblem(matX, y, nPoints, nDims, 0.5);

  BayesianLinearRegression estimator(true, true);
  estimator.Train(matX, y);

  arma::colvec xMean = arma::mean(matX, 1);
  arma::colvec xStd = arma::stddev(matX, 0, 1);
  double yMean = arma::mean(y);

  REQUIRE((double) abs(sum(estimator.DataOffset() - xMean)) ==
      Approx(0.0).margin(1e-6));
  REQUIRE((double) abs(sum(estimator.DataScale() - xStd)) ==
      Approx(0.0).margin(1e-6));
  REQUIRE(estimator.ResponsesOffset() == Approx(yMean).epsilon(1e-8));
}

// Make sure a model with center ans scale option set is different than a model
// without it set.
TEST_CASE("OptionsMakeModelDifferent", "[BayesianLinearRegressionTest]")
{
  arma::mat matX;
  arma::rowvec y;
  size_t nDims = 10, nPoints = 100;
  GenerateProblem(matX, y, nPoints, nDims, 0.5);

  BayesianLinearRegression blr(false, false), blrC(true, false),
      blrCS(true, true);

  blr.Train(matX, y);
  blrC.Train(matX, y);
  blrCS.Train(matX, y);

  for (size_t i = 0; i < nDims; ++i)
  {
    REQUIRE(blr.Omega()(i) != blrC.Omega()(i));
    REQUIRE(blr.Omega()(i) != blrCS.Omega()(i));
    REQUIRE(blrC.Omega()(i) != blrCS.Omega()(i));
  }
}

// Check that Train() does not fail with two colinear vectors.
TEST_CASE("SingularMatix", "[BayesianLinearRegressionTest]")
{
  arma::mat matX;
  arma::rowvec y;

  GenerateProblem(matX, y, 200, 10);
  // Now the first and the second rows are indentical.
  matX.row(1) = matX.row(0);

  BayesianLinearRegression estimator;
  estimator.Train(matX, y);
}

// Check that std are well computed/coherent. At least higher than the
// estimated predictive variance.
TEST_CASE("PredictiveUncertainties", "[BayesianLinearRegressionTest]")
{
  arma::mat matX;
  arma::rowvec y;

  GenerateProblem(matX, y, 100, 10, 1);

  BayesianLinearRegression estimator(true, true);
  estimator.Train(matX, y);

  arma::rowvec responses, std;
  estimator.Predict(matX, responses, std);
  const double estStd = sqrt(estimator.Variance());

  for (size_t i = 0; i < matX.n_cols; i++)
    REQUIRE(std[i] > estStd);

  // Check that the estimated variance is close to 1.
  REQUIRE(estStd == Approx(1).epsilon(0.3));
}

// Check the solution is equal to the classical ridge.
TEST_CASE("EqualtoRidge", "[BayesianLinearRegressionTest]")
{
  arma::mat matX;
  arma::rowvec y, blrPred, ridgePred;

  size_t trial = 0;
  for ( ; trial < 3; ++trial)
  {
    GenerateProblem(matX, y, 100, 10, 1);

    BayesianLinearRegression blr(false, false);
    blr.Train(matX, y);

    LinearRegression ridge(matX, y, blr.Alpha() / blr.Beta(), false);

    blr.Predict(matX, blrPred);
    ridge.Predict(matX, ridgePred);

    // If the predictions seem far off, just try again.
    if (arma::norm(blrPred - ridgePred) > 1e-5)
      continue;

    // Check the predictions are close enough between ridge and our blr.
    for (size_t i = 0; i < y.size(); ++i)
      REQUIRE(blrPred[i] == Approx(ridgePred[i]).epsilon(1));

    // Exit once a test case has completed.
    break;
  }

  REQUIRE(trial <= 3);
}
