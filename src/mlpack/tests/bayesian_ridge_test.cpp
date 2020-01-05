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
  arma::colvec noise = arma::randn(nPoints) * sigma;
  // Compute y and add noise. 
  y = omega.t() * X + noise.t();
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

  // To be neutral data_offset must be all 0.
  BOOST_REQUIRE(sum(estimator.DataOffset()) == 0.0);

  // To be neutral responses_offset must be 0.
  BOOST_REQUIRE(estimator.ResponsesOffset() == 0);

  // To be neutral data_scale must be all 1.
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

  BOOST_REQUIRE_SMALL((double) abs(sum(estimator.DataOffset() - xMean)),
                      1e-6);

  BOOST_REQUIRE_SMALL((double) abs(estimator.ResponsesOffset() - yMean),
                      1e-6);

  BOOST_REQUIRE_SMALL((double) abs(sum(estimator.DataScale() - xStd)),
                      1e-6);
}



// Check that Train() return -1 if X is singular.
BOOST_AUTO_TEST_CASE(SingularMatix)
{
  arma::mat X;
  arma::rowvec y;

  GenerateProblem(X, y, 200, 10);
  // Now the first and the second rows are indentical.
  X.row(1) = X.row(0);

  BayesianRidge estimator(false, false);
  double singular = estimator.Train(X, y);
  BOOST_REQUIRE(singular == -1);
  
  

}

BOOST_AUTO_TEST_SUITE_END();
