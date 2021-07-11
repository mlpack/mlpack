/**
 * @file tests/persistence_test.cpp
 * @author Rishabh Garg
 *
 * Test the Persistence Model for time series.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/time_series_models/persistence.hpp>

#include "test_catch_tools.hpp"
#include "catch.hpp"

using namespace mlpack;
using namespace mlpack::ts;

/**
 * Checks predictions when a vector input is given.
 */
TEST_CASE("PredictMethodTestVector1", "[PersistenceModelTest]")
{
  arma::rowvec input(10, arma::fill::randu);
  arma::rowvec predictions(10);

  PersistenceModel model;
  model.Predict(input, predictions);
  
  // Starting from i = 1 because first value is NaN which can't be tested for
  // equality.
  for(int i = 1; i < 10; i++)
  {
    REQUIRE(predictions(i) - input(i-1) == Approx(0.0).margin(1e-5));
  }
}

/**
 * Checks test predictions when train and test vectors are given.
 */
TEST_CASE("PredictMethodTestVector2", "[PersistenceModelTest]")
{
  arma::colvec train(10, arma::fill::randu);
  arma::colvec test(5, arma::fill::randu);
  arma::colvec predictions(5);

  PersistenceModel model;
  model.Predict(train, test, predictions);
  
  // First test prediction is equal to the last value of train.
  REQUIRE(predictions(0) - train(9) == Approx(0.0).margin(1e-5));
  // Looping for remaining values.
  for(int i = 1; i < 5; i++)
  {
    REQUIRE(predictions(i) - test(i-1) == Approx(0.0).margin(1e-5));
  }
}

/**
 * Checks predictions when a matrix input is given.
 */
TEST_CASE("PredictMethodTestMatrix1", "[PersistenceModelTest]")
{
  arma::mat input(10, 3, arma::fill::randu);
  arma::rowvec predictions(10);

  PersistenceModel model;
  model.Predict(input, predictions);
  
  // Starting from i = 1 because first value is NaN which can't be tested for
  // equality.
  for(int i = 1; i < 10; i++)
  {
    REQUIRE(predictions(i) - input(i-1, 2) == Approx(0.0).margin(1e-5));
  }
}

/**
 * Checks test predictions when train and test matries are given.
 */
TEST_CASE("PredictMethodTestMatrix2", "[PersistenceModelTest]")
{
  arma::mat train(10, 3, arma::fill::randu);
  arma::mat test(5, 3, arma::fill::randu);
  arma::rowvec predictions(5);

  PersistenceModel model;
  model.Predict(train, test, predictions);
  
  // First test prediction is equal to the last value of train.
  REQUIRE(predictions(0) - train(9, 2) == Approx(0.0).margin(1e-5));
  // Looping for remaining values.
  for(int i = 1; i < 5; i++)
  {
    REQUIRE(predictions(i) - test(i-1, 2) == Approx(0.0).margin(1e-5));
  }
}
