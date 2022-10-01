/**
 * @file tests/linear_regression_test.cpp
 *
 * Test for linear regression.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/linear_regression.hpp>

#include "serialization.hpp"
#include "test_catch_tools.hpp"
#include "catch.hpp"

using namespace mlpack;

/**
 * Creates two 10x3 random matrices and one 10x1 "results" matrix.
 * Finds B in y=BX with one matrix, then predicts against the other.
 */
TEST_CASE("LinearRegressionTestCase", "[LinearRegressionTest]")
{
  // Predictors and points are 10x3 matrices.
  arma::mat predictors(3, 10);
  arma::mat points(3, 10);

  // Responses is the "correct" value for each point in predictors and points.
  arma::rowvec responses(10);

  // The values we get back when we predict for points.
  arma::rowvec predictions(10);

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
    points[elem] += Random() / 50.0;
    predictors[elem] += Random() / 50.0;
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
    REQUIRE(predictions(i) - responses(i) == Approx(0.0).margin(0.05));
}

/**
 * Check the functionality of ComputeError().
 */
TEST_CASE("ComputeErrorTest", "[LinearRegressionTest]")
{
  arma::mat predictors;
  predictors = { {  0, 1, 2, 4, 8, 16 },
                 { 16, 8, 4, 2, 1,  0 } };
  arma::rowvec responses = "0 2 4 3 8 8";

  // http://www.mlpack.org/trac/ticket/298
  // This dataset gives a cost of 1.189500337 (as calculated in Octave).
  LinearRegression lr(predictors, responses);

  REQUIRE(lr.ComputeError(predictors, responses) ==
      Approx(1.189500337).epsilon(1e-5));
}

/**
 * Ensure that the cost is 0 when a perfectly-fitting dataset is given.
 */
TEST_CASE("ComputeErrorPerfectFitTest", "[LinearRegressionTest]")
{
  // Linear regression should perfectly model this dataset.
  arma::mat predictors;
  predictors = { { 0, 1, 2, 1, 6, 2 },
                 { 0, 1, 2, 2, 2, 6 } };
  arma::rowvec responses = "0 2 4 3 8 8";

  LinearRegression lr(predictors, responses);

  REQUIRE(lr.ComputeError(predictors, responses) == Approx(0.0).margin(1e-25));
}

/**
 * Test ridge regression using an empty dataset, which is not invertible.  But
 * the ridge regression part should make it invertible.
 */
TEST_CASE("RidgeRegressionTest", "[LinearRegressionTest]")
{
  // Create empty dataset.
  arma::mat data;
  data.zeros(10, 5000); // 10-dimensional, 5000 points.
  arma::rowvec responses;
  responses.zeros(5000); // 5000 points.

  // Any lambda greater than 0 works to make the predictors covariance matrix
  // invertible.  If ridge regression is not working correctly, then the matrix
  // will not be invertible and the test should segfault (or something else
  // ugly).
  LinearRegression lr(data, responses, 0.0001);

  // Now just make sure that it predicts some more zeros.
  arma::rowvec predictedResponses;
  lr.Predict(data, predictedResponses);

  for (size_t i = 0; i < 5000; ++i)
    REQUIRE((double) predictedResponses[i] == Approx(0.0).margin(1e-20));
}

/**
 * Creates two 10x3 random matrices and one 10x1 "results" matrix.
 * Finds B in y=BX with one matrix, then predicts against the other, but uses
 * ridge regression with an extremely small lambda value.
 */
TEST_CASE("RidgeRegressionTestCase", "[LinearRegressionTest]")
{
  // Predictors and points are 10x3 matrices.
  arma::mat predictors(3, 10);
  arma::mat points(3, 10);

  // Responses is the "correct" value for each point in predictors and points.
  arma::rowvec responses(10);

  // The values we get back when we predict for points.
  arma::rowvec predictions(10);

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
    points[elem] += Random() / 50.0;
    predictors[elem] += Random() / 50.0;
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
    REQUIRE(predictions(i) - responses(i) == Approx(0.0).margin(0.05));
}

/**
 * Test that a LinearRegression model trained in the constructor and trained in
 * the Train() method give the same model.
 */
TEST_CASE("LinearRegressionTrainTest", "[LinearRegressionTest]")
{
  // Random dataset.
  arma::mat dataset = arma::randu<arma::mat>(5, 1000);
  arma::rowvec responses = arma::randu<arma::rowvec>(1000);

  LinearRegression lr(dataset, responses, 0.3);
  LinearRegression lrTrain;
  lrTrain.Lambda() = 0.3;

  lrTrain.Train(dataset, responses);

  REQUIRE(lr.Parameters().n_elem == lrTrain.Parameters().n_elem);
  for (size_t i = 0; i < lr.Parameters().n_elem; ++i)
    REQUIRE(lr.Parameters()[i] ==
        Approx(lrTrain.Parameters()[i]).epsilon(1e-7));
}

/*
 * Linear regression serialization test.
 */
TEST_CASE("LinearRegressionTest", "[LinearRegressionTest]")
{
  // Generate some random data.
  arma::mat data;
  data.randn(15, 800);
  arma::rowvec responses;
  responses.randn(800);

  LinearRegression lr(data, responses, 0.05); // Train the model.
  LinearRegression xmlLr, jsonLr, binaryLr;

  SerializeObjectAll(lr, xmlLr, jsonLr, binaryLr);

  REQUIRE(lr.Lambda() == Approx(xmlLr.Lambda()).epsilon(1e-10));
  REQUIRE(lr.Lambda() == Approx(jsonLr.Lambda()).epsilon(1e-10));
  REQUIRE(lr.Lambda() == Approx(binaryLr.Lambda()).epsilon(1e-10));

  CheckMatrices(lr.Parameters(), xmlLr.Parameters(), jsonLr.Parameters(),
      binaryLr.Parameters());
}

/**
 * Test that LinearRegression::Train() returns finite OLS error.
 */
TEST_CASE("LinearRegressionTrainReturnObjective", "[LinearRegressionTest]")
{
  arma::mat predictors(3, 10);
  arma::mat points(3, 10);

  // Responses is the "correct" value for each point in predictors and points.
  arma::rowvec responses(10);

  // The values we get back when we predict for points.
  arma::rowvec predictions(10);

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
    points[elem] += Random() / 50.0;
    predictors[elem] += Random() / 50.0;
  }

  // Generate responses.
  for (size_t elem = 0; elem < responses.n_elem; elem++)
    responses[elem] = coeffs[0] +
        dot(coeffs.rows(1, 3), arma::ones<arma::rowvec>(3) * elem);

  // Initialize and predict.
  LinearRegression lr;
  double error = lr.Train(predictors, responses);

  REQUIRE(std::isfinite(error) == true);
}
