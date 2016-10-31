/**
 * @file rnn_impl.hpp
 * @author Marcus Edel
 *
 * Definition of the RNN class, which implements recurrent neural networks.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_RNN_IMPL_HPP
#define MLPACK_METHODS_ANN_RNN_IMPL_HPP

// In case it hasn't been included yet.
#include "rnn.hpp"

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
RNN<LayerTypes, OutputLayerType, InitializationRuleType, PerformanceFunction
>::RNN(LayerType &&network,
       OutputType &&outputLayer,
       const arma::mat& predictors,
       const arma::mat& responses,
       OptimizerType<NetworkType>& optimizer,
       InitializationRuleType initializeRule,
       PerformanceFunction performanceFunction) :
    network(std::forward<LayerType>(network)),
    outputLayer(std::forward<OutputType>(outputLayer)),
    performanceFunc(std::move(performanceFunction)),
    predictors(predictors),
    responses(responses),
    numFunctions(predictors.n_cols),
    inputSize(0),
    outputSize(0)
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
  Timer::Start("rnn_optimization");
  const double out = optimizer.Optimize(parameter);
  Timer::Stop("rnn_optimization");

  Log::Info << "RNN::RNN(): final objective of trained model is " << out
      << "." << std::endl;
}

template<typename LayerTypes,
         typename OutputLayerType,
         typename InitializationRuleType,
         typename PerformanceFunction
>
template<typename LayerType, typename OutputType>
RNN<LayerTypes, OutputLayerType, InitializationRuleType, PerformanceFunction
>::RNN(LayerType &&network,
       OutputType &&outputLayer,
       const arma::mat& predictors,
       const arma::mat& responses,
       InitializationRuleType initializeRule,
       PerformanceFunction performanceFunction) :
    network(std::forward<LayerType>(network)),
    outputLayer(std::forward<OutputType>(outputLayer)),
    performanceFunc(std::move(performanceFunction)),
    inputSize(0),
    outputSize(0)
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
RNN<LayerTypes, OutputLayerType, InitializationRuleType, PerformanceFunction
>::RNN(LayerType &&network,
       OutputType &&outputLayer,
       InitializationRuleType initializeRule,
       PerformanceFunction performanceFunction) :
    network(std::forward<LayerType>(network)),
    outputLayer(std::forward<OutputType>(outputLayer)),
    performanceFunc(std::move(performanceFunction)),
    inputSize(0),
    outputSize(0)
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
void RNN<
LayerTypes, OutputLayerType, InitializationRuleType, PerformanceFunction
>::Train(const arma::mat& predictors, const arma::mat& responses)
{
  numFunctions = predictors.n_cols;
  this->predictors = predictors;
  this->responses = responses;

  OptimizerType<decltype(*this)> optimizer(*this);

  // Train the model.
  Timer::Start("rnn_optimization");
  const double out = optimizer.Optimize(parameter);
  Timer::Stop("rnn_optimization");

  Log::Info << "RNN::RNN(): final objective of trained model is " << out
      << "." << std::endl;
}

template<typename LayerTypes,
         typename OutputLayerType,
         typename InitializationRuleType,
         typename PerformanceFunction
>
template<template<typename> class OptimizerType>
void RNN<
LayerTypes, OutputLayerType, InitializationRuleType, PerformanceFunction
>::Train(const arma::mat& predictors,
         const arma::mat& responses,
         OptimizerType<NetworkType>& optimizer)
{
  numFunctions = predictors.n_cols;
  this->predictors = predictors;
  this->responses = responses;

  // Train the model.
  Timer::Start("rnn_optimization");
  const double out = optimizer.Optimize(parameter);
  Timer::Stop("rnn_optimization");

  Log::Info << "RNN::RNN(): final objective of trained model is " << out
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
void RNN<
LayerTypes, OutputLayerType, InitializationRuleType, PerformanceFunction
>::Train(OptimizerType<NetworkType>& optimizer)
{
  // Train the model.
  Timer::Start("rnn_optimization");
  const double out = optimizer.Optimize(parameter);
  Timer::Stop("rnn_optimization");

  Log::Info << "RNN::RNN(): final objective of trained model is " << out
      << "." << std::endl;
}

template<typename LayerTypes,
         typename OutputLayerType,
         typename InitializationRuleType,
         typename PerformanceFunction
>
void RNN<
LayerTypes, OutputLayerType, InitializationRuleType, PerformanceFunction
>::Predict(arma::mat& predictors, arma::mat& responses)
{
  arma::mat responsesTemp;
  SinglePredict(arma::mat(predictors.colptr(0), predictors.n_rows,
      1, false, true), responsesTemp);

  responses = arma::mat(responsesTemp.n_elem, predictors.n_cols);
  responses.col(0) = responsesTemp.col(0);

  for (size_t i = 1; i < predictors.n_cols; i++)
  {
    SinglePredict(arma::mat(predictors.colptr(i), predictors.n_rows,
      1, false, true), responsesTemp);
    responses.col(i) = responsesTemp.col(0);
  }
}

template<typename LayerTypes,
         typename OutputLayerType,
         typename InitializationRuleType,
         typename PerformanceFunction
>
double RNN<
LayerTypes, OutputLayerType, InitializationRuleType, PerformanceFunction
>::Evaluate(const arma::mat& /* unused */,
            const size_t i,
            const bool deterministic)
{
  this->deterministic = deterministic;

  arma::mat input = arma::mat(predictors.colptr(i), predictors.n_rows,
      1, false, true);
  arma::mat target = arma::mat(responses.colptr(i), responses.n_rows,
      1, false, true);

  // Initialize the activation storage only once.
  if (activations.empty())
    InitLayer(input, target, network);

  double networkError = 0;
  seqLen = input.n_rows / inputSize;
  ResetParameter(network);

  error = arma::mat(outputSize, outputSize < target.n_elem ? seqLen : 1);

  // Iterate through the input sequence and perform the feed forward pass.
  for (seqNum = 0; seqNum < seqLen; seqNum++)
  {
    // Perform the forward pass and save the activations.
    Forward(input.rows(seqNum * inputSize, (seqNum + 1) * inputSize - 1),
        network);
    SaveActivations(network);

    // Retrieve output error of the subsequence.
    if (seqOutput)
    {
      arma::mat seqError = error.unsafe_col(seqNum);
      arma::mat seqTarget = target.submat(seqNum * outputSize, 0,
          (seqNum + 1) * outputSize - 1, 0);
      networkError += OutputError(seqTarget, seqError, network);
    }
  }

  // Retrieve output error of the complete sequence.
  if (!seqOutput)
    return OutputError(target, error, network);

  return networkError;
}

template<typename LayerTypes,
         typename OutputLayerType,
         typename InitializationRuleType,
         typename PerformanceFunction
>
void RNN<
LayerTypes, OutputLayerType, InitializationRuleType, PerformanceFunction
>::Gradient(const arma::mat& /* unused */,
            const size_t i,
            arma::mat& gradient)
{
  if (gradient.is_empty())
  {
    gradient = arma::zeros<arma::mat>(parameter.n_rows, parameter.n_cols);
  }
  else
  {
    gradient.zeros();
  }

  Evaluate(parameter, i, false);

  arma::mat currentGradient = arma::mat(gradient.n_rows, gradient.n_cols);
  NetworkGradients(currentGradient, network);

  const arma::mat input = arma::mat(predictors.colptr(i), predictors.n_rows,
      1, false, true);

  // Iterate through the input sequence and perform the feed backward pass.
  for (seqNum = seqLen - 1; seqNum >= 0; seqNum--)
  {
    // Load the network activation for the upcoming backward pass.
    LoadActivations(input.rows(seqNum * inputSize, (seqNum + 1) *
        inputSize - 1), network);

    // Perform the backward pass.
    if (seqOutput)
    {
      arma::mat seqError = error.unsafe_col(seqNum);
      Backward(seqError, network);
    }
    else
    {
      Backward(error, network);
    }

    // Link the parameters and update the gradients.
    LinkParameter(network);
    UpdateGradients<>(network);

    // Update the overall gradient.
    gradient += currentGradient;

    if (seqNum == 0) break;
  }
}

template<typename LayerTypes,
         typename OutputLayerType,
         typename InitializationRuleType,
         typename PerformanceFunction
>
template<typename Archive>
void RNN<
LayerTypes, OutputLayerType, InitializationRuleType, PerformanceFunction
>::Serialize(Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(parameter, "parameter");

  // If we are loading, we need to initialize the weights.
  if (Archive::is_loading::value)
  {
    NetworkWeights(parameter, network);
  }
}

} // namespace ann
} // namespace mlpack

#endif
