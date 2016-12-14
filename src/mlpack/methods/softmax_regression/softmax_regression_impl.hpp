/**
 * @file softmax_regression_impl.hpp
 * @author Siddharth Agrawal
 *
 * Implementation of softmax regression.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_IMPL_HPP
#define MLPACK_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_IMPL_HPP

// In case it hasn't been included yet.
#include "softmax_regression.hpp"

namespace mlpack {
namespace regression {

template<template<typename> class OptimizerType>
SoftmaxRegression<OptimizerType>::
SoftmaxRegression(const size_t inputSize,
                  const size_t numClasses,
                  const bool fitIntercept) :
    numClasses(numClasses),
    lambda(0.0001),
    fitIntercept(fitIntercept)
{
   SoftmaxRegressionFunction::InitializeWeights(parameters,
                                                inputSize, numClasses,
                                                fitIntercept);
}

template<template<typename> class OptimizerType>
SoftmaxRegression<OptimizerType>::SoftmaxRegression(const arma::mat& data,
                                                    const arma::Row<size_t>& labels,
                                                    const size_t numClasses,
                                                    const double lambda,
                                                    const bool fitIntercept) :
    numClasses(numClasses),
    lambda(lambda),
    fitIntercept(fitIntercept)
{
  SoftmaxRegressionFunction regressor(data, labels, numClasses,
                                      lambda, fitIntercept);
  OptimizerType<SoftmaxRegressionFunction> optimizer(regressor);

  parameters = regressor.GetInitialPoint();
  Train(optimizer);
}

template<template<typename> class OptimizerType>
SoftmaxRegression<OptimizerType>::SoftmaxRegression(
    OptimizerType<SoftmaxRegressionFunction>& optimizer) :
    parameters(optimizer.Function().GetInitialPoint()),
    numClasses(optimizer.Function().NumClasses()),
    lambda(optimizer.Function().Lambda()),
    fitIntercept(optimizer.Function().FitIntercept())
{
  Train(optimizer);
}

template<template<typename> class OptimizerType>
void SoftmaxRegression<OptimizerType>::Predict(const arma::mat& testData,
                                               arma::Row<size_t>& predictions)
    const
{
  if (testData.n_rows != FeatureSize())
  {
    std::ostringstream oss;
    oss << "SoftmaxRegression::Predict(): test data has " << testData.n_rows
        << " dimensions, but model has " << FeatureSize() << "dimensions";
    throw std::invalid_argument(oss.str());
  }

  // Calculate the probabilities for each test input.
  arma::mat hypothesis, probabilities;
  if (fitIntercept)
  {
    // In order to add the intercept term, we should compute following matrix:
    //     [1; data] = arma::join_cols(ones(1, data.n_cols), data)
    //     hypothesis = arma::exp(parameters * [1; data]).
    //
    // Since the cost of join maybe high due to the copy of original data,
    // split the hypothesis computation to two components.
    hypothesis = arma::exp(
      arma::repmat(parameters.col(0), 1, testData.n_cols) +
      parameters.cols(1, parameters.n_cols - 1) * testData);
  }
  else
  {
    hypothesis = arma::exp(parameters * testData);
  }

  probabilities = hypothesis / arma::repmat(arma::sum(hypothesis, 0),
                                            numClasses, 1);

  // Prepare necessary data.
  predictions.zeros(testData.n_cols);
  double maxProbability = 0;

  // For each test input.
  for (size_t i = 0; i < testData.n_cols; i++)
  {
    // For each class.
    for (size_t j = 0; j < numClasses; j++)
    {
      // If a higher class probability is encountered, change prediction.
      if (probabilities(j, i) > maxProbability)
      {
        maxProbability = probabilities(j, i);
        predictions(i) = j;
      }
    }

    // Set maximum probability to zero for the next input.
    maxProbability = 0;
  }
}

template<template<typename> class OptimizerType>
double SoftmaxRegression<OptimizerType>::ComputeAccuracy(
    const arma::mat& testData,
    const arma::Row<size_t>& labels) const
{
  arma::Row<size_t> predictions;

  // Get predictions for the provided data.
  Predict(testData, predictions);

  // Increment count for every correctly predicted label.
  size_t count = 0;
  for (size_t i = 0; i < predictions.n_elem; i++)
    if (predictions(i) == labels(i))
      count++;

  // Return percentage accuracy.
  return (count * 100.0) / predictions.n_elem;
}

template<template<typename> class OptimizerType>
double SoftmaxRegression<OptimizerType>::Train(
    OptimizerType<SoftmaxRegressionFunction>& optimizer)
{
  // Train the model.
  Timer::Start("softmax_regression_optimization");
  const double out = optimizer.Optimize(parameters);
  Timer::Stop("softmax_regression_optimization");

  Log::Info << "SoftmaxRegression::SoftmaxRegression(): final objective of "
            << "trained model is " << out << "." << std::endl;

  return out;
}

template<template<typename> class OptimizerType>
double SoftmaxRegression<OptimizerType>::Train(const arma::mat& data,
                                               const arma::Row<size_t>& labels,
                                               const size_t numClasses)
{
  SoftmaxRegressionFunction regressor(data, labels, numClasses,
                                      lambda, fitIntercept);
  OptimizerType<SoftmaxRegressionFunction> optimizer(regressor);

  return Train(optimizer);
}

} // namespace regression
} // namespace mlpack

#endif
