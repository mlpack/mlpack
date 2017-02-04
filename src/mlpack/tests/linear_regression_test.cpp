/**
 * @file linear_regression_test.cpp
 *
 * Test for linear regression.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::regression;

BOOST_AUTO_TEST_SUITE(LinearRegressionTest);

/**
 * Creates two 10x3 random matrices and one 10x1 "results" matrix.
 * Finds B in y=BX with one matrix, then predicts against the other.
 */
BOOST_AUTO_TEST_CASE(LinearRegressionTestCase)
{
  // Predictors and points are 10x3 matrices.
  arma::mat predictors(3, 10);
  arma::mat points(3, 10);

  // Responses is the "correct" value for each point in predictors and points.
  arma::vec responses(10);

  // The values we get back when we predict for points.
  arma::vec predictions(10);

  // We'll randomly select some coefficients for the linear response.
  arma::vec coeffs;
  coeffs.randu(4);

  // Now generate each point.
  for (size_t row = 0; row < 3; row++)
    predictors.row(row) = arma::linspace<arma::rowvec>(0, 9, 10);

  points = predictors;

  // Now add a small amount of noise to each point.
  for (size_t elem = 0; elem < points.n_elem; elem++)
  {
    // Max added noise is 0.02.
    points[elem] += math::Random() / 50.0;
    predictors[elem] += math::Random() / 50.0;
  }

  // Generate responses.
  for (size_t elem = 0; elem < responses.n_elem; elem++)
    responses[elem] = coeffs[0] +
        dot(coeffs.rows(1, 3), arma::ones<arma::rowvec>(3) * elem);

  // Initialize and predict.
  LinearRegression lr(predictors, responses);
  lr.Predict(points, predictions);

  // Output result and verify we have less than 5% error from "correct" value
  // for each point.
  for (size_t i = 0; i < predictions.n_cols; ++i)
    BOOST_REQUIRE_SMALL(predictions(i) - responses(i), .05);
}

/**
 * Check the functionality of ComputeError().
 */
BOOST_AUTO_TEST_CASE(ComputeErrorTest)
{
  arma::mat predictors;
  predictors << 0 << 1 << 2 << 4 << 8 << 16 << arma::endr
             << 16 << 8 << 4 << 2 << 1 << 0 << arma::endr;
  arma::vec responses = "0 2 4 3 8 8";

  // http://www.mlpack.org/trac/ticket/298
  // This dataset gives a cost of 1.189500337 (as calculated in Octave).
  LinearRegression lr(predictors, responses);

  BOOST_REQUIRE_CLOSE(lr.ComputeError(predictors, responses), 1.189500337,
      1e-3);
}

/**
 * Ensure that the cost is 0 when a perfectly-fitting dataset is given.
 */
BOOST_AUTO_TEST_CASE(ComputeErrorPerfectFitTest)
{
  // Linear regression should perfectly model this dataset.
  arma::mat predictors;
  predictors << 0 << 1 << 2 << 1 << 6 << 2 << arma::endr
             << 0 << 1 << 2 << 2 << 2 << 6 << arma::endr;
  arma::vec responses = "0 2 4 3 8 8";

  LinearRegression lr(predictors, responses);

  BOOST_REQUIRE_SMALL(lr.ComputeError(predictors, responses), 1e-25);
}

/**
 * Test ridge regression using an empty dataset, which is not invertible.  But
 * the ridge regression part should make it invertible.
 */
BOOST_AUTO_TEST_CASE(RidgeRegressionTest)
{
  // Create empty dataset.
  arma::mat data;
  data.zeros(10, 5000); // 10-dimensional, 5000 points.
  arma::vec responses;
  responses.zeros(5000); // 5000 points.

  // Any lambda greater than 0 works to make the predictors covariance matrix
  // invertible.  If ridge regression is not working correctly, then the matrix
  // will not be invertible and the test should segfault (or something else
  // ugly).
  LinearRegression lr(data, responses, 0.0001);

  // Now just make sure that it predicts some more zeros.
  arma::vec predictedResponses;
  lr.Predict(data, predictedResponses);

  for (size_t i = 0; i < 5000; ++i)
    BOOST_REQUIRE_SMALL((double) predictedResponses[i], 1e-20);
}

/**
 * Creates two 10x3 random matrices and one 10x1 "results" matrix.
 * Finds B in y=BX with one matrix, then predicts against the other, but uses
 * ridge regression with an extremely small lambda value.
 */
BOOST_AUTO_TEST_CASE(RidgeRegressionTestCase)
{
  // Predictors and points are 10x3 matrices.
  arma::mat predictors(3, 10);
  arma::mat points(3, 10);

  // Responses is the "correct" value for each point in predictors and points.
  arma::vec responses(10);

  // The values we get back when we predict for points.
  arma::vec predictions(10);

  // We'll randomly select some coefficients for the linear response.
  arma::vec coeffs;
  coeffs.randu(4);

  // Now generate each point.
  for (size_t row = 0; row < 3; row++)
    predictors.row(row) = arma::linspace<arma::rowvec>(0, 9, 10);

  points = predictors;

  // Now add a small amount of noise to each point.
  for (size_t elem = 0; elem < points.n_elem; elem++)
  {
    // Max added noise is 0.02.
    points[elem] += math::Random() / 50.0;
    predictors[elem] += math::Random() / 50.0;
  }

  // Generate responses.
  for (size_t elem = 0; elem < responses.n_elem; elem++)
    responses[elem] = coeffs[0] +
        dot(coeffs.rows(1, 3), arma::ones<arma::rowvec>(3) * elem);

  // Initialize and predict with very small lambda.
  LinearRegression lr(predictors, responses, 0.001);
  lr.Predict(points, predictions);

  // Output result and verify we have less than 5% error from "correct" value
  // for each point.
  for (size_t i = 0; i < predictions.n_cols; ++i)
    BOOST_REQUIRE_SMALL(predictions(i) - responses(i), .05);
}

/**
 * Test that a LinearRegression model trained in the constructor and trained in
 * the Train() method give the same model.
 */
BOOST_AUTO_TEST_CASE(LinearRegressionTrainTest)
{
  // Random dataset.
  arma::mat dataset = arma::randu<arma::mat>(5, 1000);
  arma::vec responses = arma::randu<arma::vec>(1000);

  LinearRegression lr(dataset, responses, 0.3);
  LinearRegression lrTrain;
  lrTrain.Lambda() = 0.3;

  lrTrain.Train(dataset, responses);

  BOOST_REQUIRE_EQUAL(lr.Parameters().n_elem, lrTrain.Parameters().n_elem);
  for (size_t i = 0; i < lr.Parameters().n_elem; ++i)
    BOOST_REQUIRE_CLOSE(lr.Parameters()[i], lrTrain.Parameters()[i], 1e-5);
}

BOOST_AUTO_TEST_SUITE_END();
