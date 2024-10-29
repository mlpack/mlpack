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
TEMPLATE_TEST_CASE("LinearRegressionTestCase", "[LinearRegressionTest]",
    arma::fmat, arma::mat)
{
  using MatType = TestType;
  using RowType = arma::Row<typename MatType::elem_type>;
  using ColType = arma::Col<typename MatType::elem_type>;

  // Predictors and points are 10x3 matrices.
  MatType predictors(3, 10);
  MatType points(3, 10);

  // Responses is the "correct" value for each point in predictors and points.
  RowType responses(10);

  // The values we get back when we predict for points.
  RowType predictions(10);

  // We'll randomly select some coefficients for the linear response.
  ColType coeffs;
  coeffs.randu(4);

  // Now generate each point.
  for (size_t row = 0; row < 3; row++)
    predictors.row(row) = arma::linspace<RowType>(0, 9, 10);

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
  LinearRegression<MatType> lr(predictors, responses);
  lr.Predict(points, predictions);

  // Output result and verify we have less than 5% error from "correct" value
  // for each point.
  for (size_t i = 0; i < predictions.n_cols; ++i)
    REQUIRE(predictions(i) - responses(i) == Approx(0.0).margin(0.05));
}

/**
 * Check the functionality of ComputeError().
 */
TEMPLATE_TEST_CASE("ComputeErrorTest", "[LinearRegressionTest]", arma::fmat,
    arma::mat)
{
  using MatType = TestType;
  using RowType = arma::Row<typename MatType::elem_type>;

  MatType predictors;
  predictors = { {  0, 1, 2, 4, 8, 16 },
                 { 16, 8, 4, 2, 1,  0 } };
  RowType responses = "0 2 4 3 8 8";

  // http://www.mlpack.org/trac/ticket/298
  // This dataset gives a cost of 1.189500337 (as calculated in Octave).
  LinearRegression<MatType> lr(predictors, responses);

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

  LinearRegression<> lr(predictors, responses);

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
  LinearRegression<> lr(data, responses, 0.0001);

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
  LinearRegression<> lr(predictors, responses, 0.001);
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

  LinearRegression<> lr(dataset, responses, 0.3);
  LinearRegression<> lrTrain;
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

  LinearRegression<> lr(data, responses, 0.05); // Train the model.
  LinearRegression<> xmlLr, jsonLr, binaryLr;

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
  LinearRegression<> lr;
  double error = lr.Train(predictors, responses);

  REQUIRE(std::isfinite(error) == true);
}

/**
 * Make sure all versions of Train() work correctly.
 */
TEMPLATE_TEST_CASE("LinearRegressionAllTrainVersionsTest",
    "[LinearRegressionTest]", arma::fmat, arma::mat)
{
  using MatType = TestType;
  using RowType = arma::Row<typename MatType::elem_type>;

  // The data doesn't really matter for this test; mostly we want to make sure
  // that all the Train() variants work properly.
  MatType predictors;
  predictors = { {  0, 1, 2, 4, 8, 16 },
                 { 16, 8, 4, 2, 1,  0 } };
  RowType responses = "0 2 4 3 8 8";
  RowType weights = "1.0 1.1 1.2 0.8 0.9 1.0";

  LinearRegression<MatType> lr1, lr2, lr3, lr4, lr5, lr6;

  (void) lr1.Train(predictors, responses);
  (void) lr2.Train(predictors, responses, 0.1);
  (void) lr3.Train(predictors, responses, 0.2, false);
  (void) lr4.Train(predictors, responses, weights);
  (void) lr5.Train(predictors, responses, weights, 0.3);
  (void) lr6.Train(predictors, responses, weights, 0.4, false);

  // We don't care about the specifics of the trained model, but we want to just
  // make sure everything appears to be correct from the sizes and
  // hyperparameters.
  REQUIRE(lr1.Lambda() == Approx(0.0).margin(1e-10));
  REQUIRE(lr1.Intercept() == true);
  REQUIRE(lr1.Parameters().n_elem == 3);

  REQUIRE(lr2.Lambda() == Approx(0.1).margin(1e-10));
  REQUIRE(lr2.Intercept() == true);
  REQUIRE(lr2.Parameters().n_elem == 3);

  REQUIRE(lr3.Lambda() == Approx(0.2).margin(1e-10));
  REQUIRE(lr3.Intercept() == false);
  REQUIRE(lr3.Parameters().n_elem == 2);

  REQUIRE(lr4.Lambda() == Approx(0.0).margin(1e-10));
  REQUIRE(lr4.Intercept() == true);
  REQUIRE(lr4.Parameters().n_elem == 3);

  REQUIRE(lr5.Lambda() == Approx(0.3).margin(1e-10));
  REQUIRE(lr5.Intercept() == true);
  REQUIRE(lr5.Parameters().n_elem == 3);

  REQUIRE(lr6.Lambda() == Approx(0.4).margin(1e-10));
  REQUIRE(lr6.Intercept() == false);
  REQUIRE(lr6.Parameters().n_elem == 2);

  // We can also check that the weighted model is different from the unweighted
  // model.
  REQUIRE(!arma::approx_equal(lr1.Parameters(), lr4.Parameters(), "absdiff",
      1e-5));
}

/**
 * Ensure that single-point Predict() returns the same results as multi-point
 * Predict().
 */
TEMPLATE_TEST_CASE("LinearRegressionSinglePointPredictTest",
    "[LinearRegressionTest]", arma::fmat, arma::mat)
{
  using MatType = TestType;
  using RowType = arma::Row<typename MatType::elem_type>;

  MatType predictors;
  predictors = { {  0, 1, 2, 4, 8, 16 },
                 { 16, 8, 4, 2, 1,  0 } };
  RowType responses = "0 2 4 3 8 8";

  LinearRegression<MatType> lr(predictors, responses, 0.1, true);

  // Compute predictions for test points in batch.
  RowType predictions;
  lr.Predict(predictors, predictions);

  // Now compute each prediction individually.
  for (size_t i = 0; i < predictors.n_cols; ++i)
  {
    const double prediction = lr.Predict(predictors.col(i));
    REQUIRE(prediction == Approx(predictions[i]));
  }
}

// Make sure training on submatrices and subvectors works.
TEST_CASE("LinearRegressionSubmatrixTrainingTest", "[LinearRegressionTest]")
{
  // The quality of the model doesn't matter---mostly this is a compilation
  // test.
  arma::mat predictors(100, 1000, arma::fill::randu);
  arma::rowvec responses(1000, arma::fill::randu);
  arma::rowvec weights(1000, arma::fill::randu);

  LinearRegression<> lr1(predictors.cols(0, 499), responses.subvec(0, 499));
  LinearRegression<> lr2;
  lr2.Train(predictors.cols(0, 499), responses.subvec(0, 499));
  LinearRegression<> lr3(predictors.cols(0, 499), responses.subvec(0, 499),
      weights.subvec(0, 499));
  LinearRegression<> lr4;
  lr4.Train(predictors.cols(0, 499), responses.subvec(0, 499),
      weights.subvec(0, 499));

  REQUIRE(lr1.Parameters().n_elem == 101);
  REQUIRE(lr2.Parameters().n_elem == 101);
  REQUIRE(lr3.Parameters().n_elem == 101);
  REQUIRE(lr4.Parameters().n_elem == 101);

  arma::rowvec predictions;

  lr1.Predict(predictors.cols(500, 999), predictions);
  REQUIRE(predictions.n_cols == 500);

  lr2.Predict(predictors.cols(500, 999), predictions);
  REQUIRE(predictions.n_cols == 500);

  lr3.Predict(predictors.cols(500, 999), predictions);
  REQUIRE(predictions.n_cols == 500);

  lr4.Predict(predictors.cols(500, 999), predictions);
  REQUIRE(predictions.n_cols == 500);
}

// Make sure we can train on sparse data.
TEST_CASE("LinearRegressionSparseTrainingTest", "[LinearRegressionTest]")
{
  // For this test the quality of the model doesn't matter---mostly this is a
  // compilation test, but we check that the sizes of the returned predictions
  // and the sizes of the learned models are correct.

  // Generate sparse random data.
  arma::sp_mat data;
  data.sprandu(100, 5000, 0.3);

  arma::rowvec responses(5000, arma::fill::randu);

  LinearRegression<> lr(data, responses);

  REQUIRE(lr.Parameters().n_elem == 101);

  arma::rowvec predictions;
  lr.Predict(data, predictions);

  REQUIRE(predictions.n_elem == 5000);
}
