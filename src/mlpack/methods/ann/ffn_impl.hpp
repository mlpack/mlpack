/**
 * @file ffn_impl.hpp
 * @author Marcus Edel
 *
 * Definition of the FFN class, which implements feed forward neural networks.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_FFN_IMPL_HPP
#define MLPACK_METHODS_ANN_FFN_IMPL_HPP

// In case it hasn't been included yet.
#include "ffn.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {


template<typename OutputLayerType, typename InitializationRuleType>
FFN<OutputLayerType, InitializationRuleType>::FFN(
    OutputLayerType&& outputLayer, InitializationRuleType initializeRule) :
    outputLayer(std::move(outputLayer)),
    initializeRule(initializeRule),
    width(0),
    height(0),
    reset(false)
{
  /* Nothing to do here */
}

template<typename OutputLayerType, typename InitializationRuleType>
FFN<OutputLayerType, InitializationRuleType>::FFN(
    const arma::mat& predictors,
    const arma::mat& responses,
    OutputLayerType&& outputLayer,
    InitializationRuleType initializeRule) :
    outputLayer(std::move(outputLayer)),
    initializeRule(initializeRule),
    width(0),
    height(0),
    reset(false)
{
  numFunctions = responses.n_cols;

  this->predictors = std::move(predictors);
  this->responses = std::move(responses);

  this->deterministic = true;
  ResetDeterministic();

  if (!reset)
  {
    ResetParameters();
  }
}

template<typename OutputLayerType, typename InitializationRuleType>
FFN<OutputLayerType, InitializationRuleType>::~FFN()
{
  std::for_each(network.begin(), network.end(),
      boost::apply_visitor(deleteVisitor));
}

template<typename OutputLayerType, typename InitializationRuleType>
template<template<typename> class OptimizerType>
void FFN<OutputLayerType, InitializationRuleType>::Train(
      const arma::mat& predictors,
      const arma::mat& responses,
      OptimizerType<NetworkType>& optimizer)
{
  numFunctions = responses.n_cols;

  this->predictors = std::move(predictors);
  this->responses = std::move(responses);

  this->deterministic = true;
  ResetDeterministic();

  if (!reset)
  {
    ResetParameters();
  }

  // Train the model.
  Timer::Start("ffn_optimization");
  const double out = optimizer.Optimize(parameter);
  Timer::Stop("ffn_optimization");

  Log::Info << "FFN::FFN(): final objective of trained model is " << out
      << "." << std::endl;
}

template<typename OutputLayerType, typename InitializationRuleType>
template<template<typename> class OptimizerType>
void FFN<OutputLayerType, InitializationRuleType>::Train(
    const arma::mat& predictors, const arma::mat& responses)
{
  numFunctions = responses.n_cols;

  this->predictors = std::move(predictors);
  this->responses = std::move(responses);

  this->deterministic = true;
  ResetDeterministic();

  if (!reset)
  {
    ResetParameters();
  }

  OptimizerType<decltype(*this)> optimizer(*this);

  // Train the model.
  Timer::Start("ffn_optimization");
  const double out = optimizer.Optimize(parameter);
  Timer::Stop("ffn_optimization");

  Log::Info << "FFN::FFN(): final objective of trained model is " << out
      << "." << std::endl;
}

template<typename OutputLayerType, typename InitializationRuleType>
void FFN<OutputLayerType, InitializationRuleType>::Predict(
    arma::mat& predictors, arma::mat& responses)
{
  if (parameter.is_empty())
  {
    ResetParameters();
  }

  if (!deterministic)
  {
    deterministic = true;
    ResetDeterministic();
  }

  arma::mat responsesTemp;
  Forward(std::move(arma::mat(predictors.colptr(0),
      predictors.n_rows, 1, false, true)));
  responsesTemp = boost::apply_visitor(outputParameterVisitor,
      network.back()).col(0);

  responses = arma::mat(responsesTemp.n_elem, predictors.n_cols);
  responses.col(0) = responsesTemp.col(0);

  for (size_t i = 1; i < predictors.n_cols; i++)
  {
    Forward(std::move(arma::mat(predictors.colptr(i),
        predictors.n_rows, 1, false, true)));

    responsesTemp = boost::apply_visitor(outputParameterVisitor,
        network.back());
    responses.col(i) = responsesTemp.col(0);
  }
}

template<typename OutputLayerType, typename InitializationRuleType>
double FFN<OutputLayerType, InitializationRuleType>::Evaluate(
    const arma::mat& /* parameters */, const size_t i, const bool deterministic)
{
  if (parameter.is_empty())
  {
    ResetParameters();
  }

  if (deterministic != this->deterministic)
  {
    this->deterministic = deterministic;
    ResetDeterministic();
  }

  currentInput = std::move(arma::mat(predictors.colptr(i),
      predictors.n_rows, 1, false, true));

  Forward(std::move(currentInput));

  currentTarget = arma::mat(responses.colptr(i), responses.n_rows,
      1, false, true);

  double res = outputLayer.Forward(std::move(boost::apply_visitor(
      outputParameterVisitor, network.back())), std::move(currentTarget));

  return res;
}

template<typename OutputLayerType, typename InitializationRuleType>
void FFN<OutputLayerType, InitializationRuleType>::Gradient(
    const arma::mat& parameters, const size_t i, arma::mat& gradient)
{
  if (gradient.is_empty())
  {
    if (parameter.is_empty())
    {
      ResetParameters();
    }

    gradient = arma::zeros<arma::mat>(parameter.n_rows, parameter.n_cols);
  }
  else
  {
    gradient.zeros();
  }

  Evaluate(parameters, i, false);

  outputLayer.Backward(std::move(boost::apply_visitor(outputParameterVisitor,
      network.back())), std::move(currentTarget), std::move(error));

  Backward();
  ResetGradients(gradient);
  Gradient();
}

template<typename OutputLayerType, typename InitializationRuleType>
void FFN<OutputLayerType, InitializationRuleType>::ResetParameters()
{
  size_t weights = 0;
  for (size_t i = 0; i < network.size(); ++i)
  {
    weights += boost::apply_visitor(weightSizeVisitor, network[i]);
  }

  parameter.set_size(weights, 1);
  initializeRule.Initialize(parameter, parameter.n_elem, 1);

  size_t offset = 0;
  for (size_t i = 0; i < network.size(); ++i)
  {
    offset += boost::apply_visitor(WeightSetVisitor(std::move(parameter),
        offset), network[i]);

    boost::apply_visitor(resetVisitor, network[i]);
  }
}

template<typename OutputLayerType, typename InitializationRuleType>
void FFN<OutputLayerType, InitializationRuleType>::ResetDeterministic()
{
  DeterministicSetVisitor deterministicSetVisitor(deterministic);
  std::for_each(network.begin(), network.end(),
      boost::apply_visitor(deterministicSetVisitor));
}

template<typename OutputLayerType, typename InitializationRuleType>
void FFN<OutputLayerType, InitializationRuleType>::ResetGradients(
    arma::mat& gradient)
{
  size_t offset = 0;
  for (size_t i = 0; i < network.size(); ++i)
  {
    offset += boost::apply_visitor(GradientSetVisitor(std::move(gradient),
        offset), network[i]);
  }
}

template<typename OutputLayerType, typename InitializationRuleType>
void FFN<OutputLayerType, InitializationRuleType>::Forward(arma::mat&& input)
{
  boost::apply_visitor(ForwardVisitor(std::move(input), std::move(
      boost::apply_visitor(outputParameterVisitor, network.front()))),
      network.front());

  if (!reset)
  {
    if (boost::apply_visitor(outputWidthVisitor, network.front()) != 0)
    {
      width = boost::apply_visitor(outputWidthVisitor, network.front());
    }

    if (boost::apply_visitor(outputHeightVisitor, network.front()) != 0)
    {
      height = boost::apply_visitor(outputHeightVisitor, network.front());
    }
  }

  for (size_t i = 1; i < network.size(); ++i)
  {
    if (!reset)
    {
      // Set the input width.
      boost::apply_visitor(SetInputWidthVisitor(width), network[i]);

      // Set the input height.
      boost::apply_visitor(SetInputHeightVisitor(height), network[i]);
    }

    boost::apply_visitor(ForwardVisitor(std::move(boost::apply_visitor(
        outputParameterVisitor, network[i - 1])), std::move(
        boost::apply_visitor(outputParameterVisitor, network[i]))), network[i]);

    if (!reset)
    {
      // Get the output width.
      if (boost::apply_visitor(outputWidthVisitor, network[i]) != 0)
      {
        width = boost::apply_visitor(outputWidthVisitor, network[i]);
      }

      // Get the output height.
      if (boost::apply_visitor(outputHeightVisitor, network[i]) != 0)
      {
        height = boost::apply_visitor(outputHeightVisitor, network[i]);
      }
    }
  }

  if (!reset)
  {
    reset = true;
  }
}

template<typename OutputLayerType, typename InitializationRuleType>
void FFN<OutputLayerType, InitializationRuleType>::Backward()
{
  boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, network.back())), std::move(error), std::move(
      boost::apply_visitor(deltaVisitor, network.back()))), network.back());

  for (size_t i = 2; i < network.size(); ++i)
  {
    boost::apply_visitor(BackwardVisitor(std::move(boost::apply_visitor(
        outputParameterVisitor, network[network.size() - i])), std::move(
        boost::apply_visitor(deltaVisitor, network[network.size() - i + 1])),
        std::move(boost::apply_visitor(deltaVisitor,
        network[network.size() - i]))), network[network.size() - i]);
  }
}

template<typename OutputLayerType, typename InitializationRuleType>
void FFN<OutputLayerType, InitializationRuleType>::Gradient()
{
  boost::apply_visitor(GradientVisitor(std::move(currentInput), std::move(
      boost::apply_visitor(deltaVisitor, network[1]))), network.front());

  for (size_t i = 1; i < network.size() - 1; ++i)
  {
    boost::apply_visitor(GradientVisitor(std::move(boost::apply_visitor(
        outputParameterVisitor, network[i - 1])), std::move(
        boost::apply_visitor(deltaVisitor, network[i + 1]))), network[i]);
  }

  boost::apply_visitor(GradientVisitor(std::move(boost::apply_visitor(
      outputParameterVisitor, network[network.size() - 2])), std::move(error)),
      network[network.size() - 1]);
}

template<typename OutputLayerType, typename InitializationRuleType>
template<typename Archive>
void FFN<OutputLayerType, InitializationRuleType>::Serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(parameter, "parameter");
  ar & data::CreateNVP(width, "width");
  ar & data::CreateNVP(height, "height");

  // If we are loading, we need to initialize the weights.
  if (Archive::is_loading::value)
  {
    reset = false;

    size_t offset = 0;
    for (size_t i = 0; i < network.size(); ++i)
    {
      offset += boost::apply_visitor(WeightSetVisitor(std::move(parameter),
          offset), network[i]);

      boost::apply_visitor(resetVisitor, network[i]);
    }
  }
}

} // namespace ann
} // namespace mlpack

#endif
