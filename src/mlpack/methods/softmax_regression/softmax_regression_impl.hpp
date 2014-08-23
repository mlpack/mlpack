/**
 * @file softmax_regression_impl.hpp
 * @author Siddharth Agrawal
 *
 * Implementation of softmax regression.
 */
#ifndef __MLPACK_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_IMPL_HPP
#define __MLPACK_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_IMPL_HPP

// In case it hasn't been included yet.
#include "softmax_regression.hpp"

namespace mlpack {
namespace regression {

template<template<typename> class OptimizerType>
SoftmaxRegression<OptimizerType>::SoftmaxRegression(const arma::mat& data,
                                                    const arma::vec& labels,
                                                    const size_t inputSize,
                                                    const size_t numClasses,
                                                    const double lambda) :
    inputSize(inputSize),
    numClasses(numClasses),
    lambda(lambda)
{
  SoftmaxRegressionFunction regressor(data, labels, inputSize, numClasses,
                                      lambda);
  OptimizerType<SoftmaxRegressionFunction> optimizer(regressor);
  
  parameters = regressor.GetInitialPoint();

  // Train the model.
  Timer::Start("softmax_regression_optimization");
  const double out = optimizer.Optimize(parameters);
  Timer::Stop("softmax_regression_optimization");

  Log::Info << "SoftmaxRegression::SoftmaxRegression(): final objective of "
      << "trained model is " << out << "." << std::endl;
}

template<template<typename> class OptimizerType>
SoftmaxRegression<OptimizerType>::SoftmaxRegression(
    OptimizerType<SoftmaxRegressionFunction>& optimizer) :
    parameters(optimizer.Function().GetInitialPoint()),
    inputSize(optimizer.Function().InputSize()),
    numClasses(optimizer.Function().NumClasses()),
    lambda(optimizer.Function().Lambda())
{
  // Train the model.
  Timer::Start("softmax_regression_optimization");
  const double out = optimizer.Optimize(parameters);
  Timer::Stop("softmax_regression_optimization");

  Log::Info << "SoftmaxRegression::SoftmaxRegression(): final objective of "
      << "trained model is " << out << "." << std::endl;
}

template<template<typename> class OptimizerType>
void SoftmaxRegression<OptimizerType>::Predict(const arma::mat& testData,
                                               arma::vec& predictions)
{
  // Calculate the probabilities for each test input.
  arma::mat hypothesis, probabilities;
  
  hypothesis = arma::exp(parameters * testData);
  probabilities = hypothesis / arma::repmat(arma::sum(hypothesis, 0),
                                            numClasses, 1);
  
  // Prepare necessary data.
  predictions.zeros(testData.n_cols);
  double maxProbability = 0;
  
  // For each test input.
  for(size_t i = 0; i < testData.n_cols; i++)
  {
    // For each class.
    for(size_t j = 0; j < numClasses; j++)
    {
      // If a higher class probability is encountered, change prediction.
      if(probabilities(j, i) > maxProbability)
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
    const arma::vec& labels)
{
  arma::vec predictions;
  
  // Get predictions for the provided data.
  Predict(testData, predictions);
  
  // Increment count for every correctly predicted label.
  size_t count = 0;
  for(size_t i = 0; i < predictions.n_elem; i++)
    if(predictions(i) == labels(i))
      count++;
  
  // Return percentage accuracy.
  return (count * 100.0) / predictions.n_elem;
}

}; // namespace regression
}; // namespace mlpack

#endif
