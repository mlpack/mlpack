/**
 * @file cnn_impl.hpp
 * @author Marcus Edel
 *
 * Definition of the CNN class, which implements convolutional neural networks.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_CNN_IMPL_HPP
#define MLPACK_METHODS_ANN_CNN_IMPL_HPP

// In case it hasn't been included yet.
#include "cnn.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {


template<typename LayerTypes,
         typename OutputLayerType,
         typename InitializationRuleType,
         typename PerformanceFunction
>
template<typename LayerType,
         typename OutputType,
         template<typename> class OptimizerType
>
CNN<LayerTypes, OutputLayerType, InitializationRuleType, PerformanceFunction
>::CNN(LayerType &&network,
       OutputType &&outputLayer,
       const arma::cube& predictors,
       const arma::mat& responses,
       OptimizerType<NetworkType>& optimizer,
       InitializationRuleType initializeRule,
       PerformanceFunction performanceFunction) :
    network(std::forward<LayerType>(network)),
    outputLayer(std::forward<OutputType>(outputLayer)),
    performanceFunc(std::move(performanceFunction)),
    predictors(predictors),
    responses(responses),
    numFunctions(predictors.n_cols)
{
  static_assert(std::is_same<typename std::decay<LayerType>::type,
                  LayerTypes>::value,
                  "The type of network must be LayerTypes.");

  static_assert(std::is_same<typename std::decay<OutputType>::type,
                OutputLayerType>::value,
                "The type of outputLayer must be OutputLayerType.");

  initializeRule.Initialize(parameter, NetworkSize(this->network), 1);
  NetworkWeights(parameter, this->network);

  // Train the model.
  Timer::Start("cnn_optimization");
  const double out = optimizer.Optimize(parameter);
  Timer::Stop("cnn_optimization");

  Log::Info << "CNN::CNN(): final objective of trained model is " << out
      << "." << std::endl;
}

template<typename LayerTypes,
         typename OutputLayerType,
         typename InitializationRuleType,
         typename PerformanceFunction
>
template<typename LayerType, typename OutputType>
CNN<LayerTypes, OutputLayerType, InitializationRuleType, PerformanceFunction
>::CNN(LayerType &&network,
       OutputType &&outputLayer,
       const arma::cube& predictors,
       const arma::mat& responses,
       InitializationRuleType initializeRule,
       PerformanceFunction performanceFunction) :
    network(std::forward<LayerType>(network)),
    outputLayer(std::forward<OutputType>(outputLayer)),
    performanceFunc(std::move(performanceFunction))
{
  static_assert(std::is_same<typename std::decay<LayerType>::type,
                  LayerTypes>::value,
                  "The type of network must be LayerTypes.");

  static_assert(std::is_same<typename std::decay<OutputType>::type,
                OutputLayerType>::value,
                "The type of outputLayer must be OutputLayerType.");

  initializeRule.Initialize(parameter, NetworkSize(this->network), 1);
  NetworkWeights(parameter, this->network);

  Train(predictors, responses);
}

template<typename LayerTypes,
         typename OutputLayerType,
         typename InitializationRuleType,
         typename PerformanceFunction
>
template<typename LayerType, typename OutputType>
CNN<LayerTypes, OutputLayerType, InitializationRuleType, PerformanceFunction
>::CNN(LayerType &&network,
       OutputType &&outputLayer,
       InitializationRuleType initializeRule,
       PerformanceFunction performanceFunction) :
    network(std::forward<LayerType>(network)),
    outputLayer(std::forward<OutputType>(outputLayer)),
    performanceFunc(std::move(performanceFunction))
{
  static_assert(std::is_same<typename std::decay<LayerType>::type,
                  LayerTypes>::value,
                  "The type of network must be LayerTypes.");

  static_assert(std::is_same<typename std::decay<OutputType>::type,
                OutputLayerType>::value,
                "The type of outputLayer must be OutputLayerType.");

  initializeRule.Initialize(parameter, NetworkSize(this->network), 1);
  NetworkWeights(parameter, this->network);
}

template<typename LayerTypes,
         typename OutputLayerType,
         typename InitializationRuleType,
         typename PerformanceFunction
>
template<template<typename> class OptimizerType>
void CNN<
LayerTypes, OutputLayerType, InitializationRuleType, PerformanceFunction
>::Train(const arma::cube& predictors, const arma::mat& responses)
{
  numFunctions = predictors.n_cols;
  sampleSize = predictors.n_slices / responses.n_cols;
  this->predictors = predictors;
  this->responses = responses;

  OptimizerType<decltype(*this)> optimizer(*this);

  // Train the model.
  Timer::Start("cnn_optimization");
  const double out = optimizer.Optimize(parameter);
  Timer::Stop("cnn_optimization");

  Log::Info << "CNN::CNN(): final objective of trained model is " << out
      << "." << std::endl;
}

template<typename LayerTypes,
         typename OutputLayerType,
         typename InitializationRuleType,
         typename PerformanceFunction
>
template<template<typename> class OptimizerType>
void CNN<
LayerTypes, OutputLayerType, InitializationRuleType, PerformanceFunction
>::Train(const arma::cube& predictors,
         const arma::mat& responses,
         OptimizerType<NetworkType>& optimizer)
{
  numFunctions = responses.n_cols;
  sampleSize = predictors.n_slices / responses.n_cols;
  this->predictors = predictors;
  this->responses = responses;

  // Train the model.
  Timer::Start("cnn_optimization");
  const double out = optimizer.Optimize(parameter);
  Timer::Stop("cnn_optimization");

  Log::Info << "CNN::CNN(): final objective of trained model is " << out
      << "." << std::endl;
}

template<typename LayerTypes,
         typename OutputLayerType,
         typename InitializationRuleType,
         typename PerformanceFunction
>
template<
    template<typename> class OptimizerType
>
void CNN<
LayerTypes, OutputLayerType, InitializationRuleType, PerformanceFunction
>::Train(OptimizerType<NetworkType>& optimizer)
{
  // Train the model.
  Timer::Start("cnn_optimization");
  const double out = optimizer.Optimize(parameter);
  Timer::Stop("cnn_optimization");

  Log::Info << "CNN::CNN(): final objective of trained model is " << out
      << "." << std::endl;
}

template<typename LayerTypes,
         typename OutputLayerType,
         typename InitializationRuleType,
         typename PerformanceFunction
>
void CNN<
LayerTypes, OutputLayerType, InitializationRuleType, PerformanceFunction
>::Predict(arma::cube& predictors, arma::mat& responses)
{
  deterministic = true;

  arma::mat responsesTemp;
  ResetParameter(network);
  Forward(predictors.slices(0, sampleSize - 1), network);
  OutputPrediction(responsesTemp, network);

  responses = arma::mat(responsesTemp.n_elem, predictors.n_slices);
  responses.col(0) = responsesTemp.col(0);

  for (size_t i = 1; i < (predictors.n_slices / sampleSize); i++)
  {
    Forward(predictors.slices(i, (i + 1) * sampleSize - 1), network);

    responsesTemp = arma::mat(responses.colptr(i), responses.n_rows, 1, false,
        true);
    OutputPrediction(responsesTemp, network);
    responses.col(i) = responsesTemp.col(0);
  }
}

template<typename LayerTypes,
         typename OutputLayerType,
         typename InitializationRuleType,
         typename PerformanceFunction
>
double CNN<
LayerTypes, OutputLayerType, InitializationRuleType, PerformanceFunction
>::Evaluate(const arma::mat& /* unused */,
            const size_t i,
            const bool deterministic)
{
  this->deterministic = deterministic;

  ResetParameter(network);
  Forward(predictors.slices(i, (i + 1) * sampleSize - 1), network);

  return OutputError(arma::mat(responses.colptr(i), responses.n_rows, 1, false,
      true), error, network);
}

template<typename LayerTypes,
         typename OutputLayerType,
         typename InitializationRuleType,
         typename PerformanceFunction
>
void CNN<
LayerTypes, OutputLayerType, InitializationRuleType, PerformanceFunction
>::Gradient(const arma::mat& /* unused */,
            const size_t i,
            arma::mat& gradient)
{
  Evaluate(parameter, i, false);

  NetworkGradients(gradient, network);

  Backward<>(error, network);
  UpdateGradients<>(network);
}

template<typename LayerTypes,
         typename OutputLayerType,
         typename InitializationRuleType,
         typename PerformanceFunction
>
template<typename Archive>
void CNN<
LayerTypes, OutputLayerType, InitializationRuleType, PerformanceFunction
>::Serialize(Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(parameter, "parameter");
  ar & data::CreateNVP(sampleSize, "sampleSize");

  // If we are loading, we need to initialize the weights.
  if (Archive::is_loading::value)
  {
    NetworkWeights(parameter, network);
  }
}

} // namespace ann
} // namespace mlpack

#endif
