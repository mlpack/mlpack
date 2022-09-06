/**
 * @file tests/test_function_tools.hpp
 * @author Marcus Edel
 *
 * This file provides some useful test function methods.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_TESTS_TEST_FUNCTION_TOOLS_HPP
#define MLPACK_TESTS_TEST_FUNCTION_TOOLS_HPP

#include <mlpack/core.hpp>

#include <mlpack/methods/logistic_regression/logistic_regression.hpp>
#include <mlpack/core/data/split_data.hpp>

using namespace mlpack;

/**
 * Create the data for the a logistic regression test.
 *
 * @param data Matrix object to store the data into.
 * @param testData Matrix object to store the test data into.
 * @param shuffledData Matrix object to store the shuffled data into.
 * @param responses Matrix object to store the overall responses into.
 * @param testResponses Matrix object to store the test responses into.
 * @param shuffledResponses Matrix object to store the shuffled responses into.
 */
inline void LogisticRegressionTestData(arma::mat& data,
                                arma::mat& testData,
                                arma::mat& shuffledData,
                                arma::Row<size_t>& responses,
                                arma::Row<size_t>& testResponses,
                                arma::Row<size_t>& shuffledResponses)
{
  // Generate a two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 1.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("9.0 9.0 9.0"), arma::eye<arma::mat>(3, 3));

  data = arma::mat(3, 1000);
  responses = arma::Row<size_t>(1000);
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
  shuffledData = arma::mat(3, 1000);
  shuffledResponses = arma::Row<size_t>(1000);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    shuffledData.col(i) = data.col(indices[i]);
    shuffledResponses[i] = responses[indices[i]];
  }

  // Create a test set.
  testData = arma::mat(3, 1000);
  testResponses = arma::Row<size_t>(1000);
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
}

template<typename MatType>
void LoadBostonHousingDataset(MatType& trainData,
                              MatType& testData,
                              arma::rowvec& trainResponses,
                              arma::rowvec& testResponses,
                              data::DatasetInfo& info)
{
  MatType dataset;
  arma::rowvec responses;

  // Defining categorical deimensions.
  info.SetDimensionality(13);
  info.Type(3) = data::Datatype::categorical;
  info.Type(8) = data::Datatype::categorical;

  if (!data::Load("boston_housing_price.csv", dataset, info))
    FAIL("Cannot load test dataset boston_housing_price.csv!");
  if (!data::Load("boston_housing_price_responses.csv", responses))
    FAIL("Cannot load test dataset boston_housing_price_responses.csv!");

  data::Split(dataset, responses, trainData, testData,
      trainResponses, testResponses, 0.3);
}

inline double RMSE(const arma::Row<double>& predictions,
                   const arma::Row<double>& trueResponses)
{
  double mse = arma::accu(arma::square(predictions - trueResponses)) /
      predictions.n_elem;
  return sqrt(mse);
}

#endif
