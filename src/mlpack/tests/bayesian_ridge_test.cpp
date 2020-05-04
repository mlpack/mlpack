/**
 * @file bayesian_ridge_test.cpp
 * @author Clement Mercier
 *
 * Test for BayesianRidge.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core/data/load.hpp>
#include <mlpack/methods/bayesian_ridge/bayesian_ridge.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>

#include <boost/test/unit_test.hpp>

using namespace mlpack::regression;
using namespace mlpack::data;

BOOST_AUTO_TEST_SUITE(BayesianRidgeTest);

void GenerateProblem(arma::mat& X,
                     arma::rowvec& y,
                     size_t nPoints,
                     size_t nDims,
                     float sigma = 0.0)
{
  X = arma::randn(nDims, nPoints);
  arma::colvec omega = arma::randn(nDims);
  // Compute y and add noise.
  y = omega.t() * X + arma::randn(nPoints).t() * sigma;
}

// Ensure that predictions are close enough to the target
// for a free noise dataset.
BOOST_AUTO_TEST_CASE(BayesianRidgeRegressionTest)
{
  arma::mat X;
  arma::rowvec y, predictions;

  GenerateProblem(X, y, 200, 10);

  // Instanciate and train the estimator.
  BayesianRidge estimator(true);
  estimator.Train(X, y);
  estimator.Predict(X, predictions);

  // Check the predictions are close enough to the targets in a free noise case.
  for (size_t i = 0; i < y.size(); i++)
    BOOST_REQUIRE_CLOSE(predictions[i], y[i], 1e-6);

  // Check that the estimated variance is zero.
  BOOST_REQUIRE_SMALL(estimator.Variance(), 1e-6);
}

// Verify fitIntercept and normalize equal false do not affect the solution.
BOOST_AUTO_TEST_CASE(TestCenter0Normalize0)
{
  arma::mat X;
  arma::rowvec y;
  size_t nDims = 30, nPoints = 100;

  GenerateProblem(X, y, nPoints, nDims, 0.5);

  BayesianRidge estimator(false, false);

  estimator.Train(X, y);

  // To be neutral dataOffset must be all 0.
  BOOST_REQUIRE(sum(estimator.DataOffset()) == 0.0);

  // To be neutral responseOffset must be 0.
  BOOST_REQUIRE(estimator.ResponsesOffset() == 0);

  // To be neutral dataScale must be all 1.
  BOOST_REQUIRE(sum(estimator.DataScale()) == nDims);
}

// Verify that centering and normalization are correct.
BOOST_AUTO_TEST_CASE(TestCenter1Normalize1)
{
  arma::mat X;
  arma::rowvec y;
  size_t nDims = 30, nPoints = 100;
  GenerateProblem(X, y, nPoints, nDims, 0.5);

  BayesianRidge estimator(true, true);
  estimator.Train(X, y);

  arma::colvec xMean = arma::mean(X, 1);
  arma::colvec xStd = arma::stddev(X, 0, 1);
  double yMean = arma::mean(y);

  BOOST_REQUIRE_SMALL((double) abs(sum(estimator.DataOffset() - xMean)), 1e-6);
  BOOST_REQUIRE_SMALL((double) abs(estimator.ResponsesOffset() - yMean), 1e-6);
  BOOST_REQUIRE_SMALL((double) abs(sum(estimator.DataScale() - xStd)), 1e-6);
}

// Check that Train() does not fail with two colinear vectors.
BOOST_AUTO_TEST_CASE(SingularMatix)
{
  arma::mat X;
  arma::rowvec y;

  GenerateProblem(X, y, 200, 10);
  // Now the first and the second rows are indentical.
  X.row(1) = X.row(0);

  BayesianRidge estimator;
  double singular = estimator.Train(X, y);
  BOOST_REQUIRE(singular != -1);
}

// Check that std are well computed/coherent. At least higher than the
// estimated predictive variance.
BOOST_AUTO_TEST_CASE(PredictiveUncertainties)
{
  arma::mat X;
  arma::rowvec y;

  GenerateProblem(X, y, 100, 10, 1);

  BayesianRidge estimator(true, true);
  estimator.Train(X, y);

  arma::rowvec responses, std;
  estimator.Predict(X, responses, std);
  const double estStd = sqrt(estimator.Variance());

  for (size_t i = 0; i < X.n_cols; i++)
    BOOST_REQUIRE(std[i] > estStd);
}

// Check the solution is equal to the classical ridge.
BOOST_AUTO_TEST_CASE(EqualtoRidge)
{
  arma::mat X;
  arma::rowvec y;

  GenerateProblem(X, y, 100, 10, 1);

  BayesianRidge bayesRidge(false, false);
  bayesRidge.Train(X, y);

  LinearRegression classicalRidge(X,
				  y,
				  bayesRidge.Alpha() / bayesRidge.Beta(),
				  false);
  double equalSol = arma::sum(bayesRidge.Omega() - classicalRidge.Parameters());
  BOOST_REQUIRE(equalSol < 1e-5);
}

BOOST_AUTO_TEST_SUITE_END();
