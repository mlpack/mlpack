#include <mlpack/core.h>
#include <mlpack/methods/linear_regression/linear_regression.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(LinearRegressonTest);

  /**
   * Creates two 10x3 random matrices and one 10x1 "results" matrix.
   * Finds B in y=BX with one matrix, then predicts against the other.
   */
  BOOST_AUTO_TEST_CASE(LinearRegressionTest)
  {
    // predictors, points are 10x3 matrices
    arma::mat predictors, points;

    // responses is the "correct" value for each point in predictors, poitns
    arma::colvec responses;

    // the values we get back when we predict for points
    arma::rowvec predictions;

    // Initialize randomly
    predictors.randu(3,10);
    points.randu(3,10);
    // add 3 so that we have two clusters of points
    predictors.cols(0,4) += 3;
    points.cols(0,4) += 3;

    // Create y
    responses.zeros(10);
    // Create a second "class" for the first cluster of points
    for(size_t i = 0; i < 5; ++i)
    {
      responses(i) = 1;
    }
    responses += 1; // "classes" are 2,1

    predictions.zeros(responses.n_rows);

    // Initialize and predict
    mlpack::linear_regression::LinearRegression lr(predictors, responses);
    lr.predict(predictions, points);

    // Output result and verify we have less than .5 error from "correct" value
    // for each point
    std::cout << points << '\n' << predictions << '\n';
    for(size_t i = 0; i < predictions.n_cols; ++i)
    {
      assert( fabs(predictions(i) - responses(i)) < .5);
    }
  }

BOOST_AUTO_TEST_SUITE_END();
