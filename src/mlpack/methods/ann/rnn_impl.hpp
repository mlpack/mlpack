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


template<typename OutputLayerType, typename InitializationRuleType>
RNN<OutputLayerType,
    InitializationRuleType
>::RNN(const size_t rho,
       const bool single,
       OutputLayerType outputLayer,
       InitializationRuleType initializeRule) :
    rho(rho),
    outputLayer(outputLayer),
    initializeRule(initializeRule),
    inputSize(0),
    outputSize(0),
    targetSize(0),
    reset(false),
    single(single)
{
  /* Nothing to do here */
}

template<typename OutputLayerType, typename InitializationRuleType>
RNN<OutputLayerType,
    InitializationRuleType
>::RNN(const arma::mat& predictors,
       const arma::mat& responses,
       const size_t rho,
       const bool single,
       OutputLayerType outputLayer,
       InitializationRuleType initializeRule) :
    rho(rho),
    outputLayer(outputLayer),
    initializeRule(initializeRule),
    inputSize(0),
    outputSize(0),
    targetSize(0),
    reset(false),
    single(single)
{
  numFunctions = responses.n_cols;

  this->predictors = std::move(predictors);
  this->responses = std::move(responses);

  this->deterministic = true;
  ResetDeterministic();

  if (!reset)
  {
    ResetParameters();
    reset = true;
  }
}

template<typename OutputLayerType, typename InitializationRuleType>
RNN<OutputLayerType,
     InitializationRuleType
>::~RNN()
{
  for (LayerTypes& layer : network)
  {
    boost::apply_visitor(deleteVisitor, layer);
  }
}

template<typename OutputLayerType, typename InitializationRuleType>
template<template<typename> class OptimizerType>
void RNN<OutputLayerType,
         InitializationRuleType
>::Train(const arma::mat& predictors,
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
    reset = true;
  }

  // Train the model.
  Timer::Start("rnn_optimization");
  const double out = optimizer.Optimize(parameter);
  Timer::Stop("rnn_optimization");

  Log::Info << "RNN::RNN(): final objective of trained model is " << out
      << "." << std::endl;
}

template<typename OutputLayerType, typename InitializationRuleType>
void RNN<OutputLayerType,
         InitializationRuleType
>::Predict(arma::mat& predictors, arma::mat& responses)
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

  responses = arma::zeros<arma::mat>(outputSize * rho, predictors.n_cols);
  arma::mat responsesTemp = responses.col(0);

  for (size_t i = 0; i < predictors.n_cols; i++)
  {
    SinglePredict(
        arma::mat(predictors.colptr(i), predictors.n_rows, 1, false, true),
        responsesTemp);

    responses.col(i) = responsesTemp;
  }
}

template<typename OutputLayerType, typename InitializationRuleType>
void RNN<OutputLayerType,
         InitializationRuleType
>::SinglePredict(const arma::mat& predictors, arma::mat& responses)
{
  for (size_t seqNum = 0; seqNum < rho; ++seqNum)
  {
    currentInput = predictors.rows(seqNum * inputSize,
        (seqNum + 1) * inputSize - 1);
    Forward(std::move(currentInput));

    responses.rows(seqNum * outputSize, (seqNum + 1) * outputSize - 1) =
        boost::apply_visitor(outputParameterVisitor, network.back());
  }
}

template<typename OutputLayerType, typename InitializationRuleType>
double RNN<OutputLayerType,
         InitializationRuleType
>::Evaluate(const arma::mat& /* parameters */,
            const size_t i,
            const bool deterministic)
{
  if (parameter.is_empty())
  {
    ResetParameters();
    reset = true;
  }

  if (deterministic != this->deterministic)
  {
    this->deterministic = deterministic;
    ResetDeterministic();
  }

  arma::mat input = arma::mat(predictors.colptr(i), predictors.n_rows,
      1, false, true);
  arma::mat target = arma::mat(responses.colptr(i), responses.n_rows,
      1, false, true);

  if (!inputSize)
  {
    inputSize = input.n_elem / rho;
    targetSize = target.n_elem / rho;
  }

  double performance = 0;

  for (size_t seqNum = 0; seqNum < rho; ++seqNum)
  {
    currentInput = input.rows(seqNum * inputSize, (seqNum + 1) * inputSize - 1);
    arma::mat currentTarget = target.rows(seqNum * targetSize,
        (seqNum + 1) * targetSize - 1);

    Forward(std::move(currentInput));

    if (!deterministic)
    {
      for (size_t l = 0; l < network.size(); ++l)
      {
        boost::apply_visitor(SaveOutputParameterVisitor(
            std::move(moduleOutputParameter)), network[l]);
      }
    }

    performance += outputLayer.Forward(std::move(boost::apply_visitor(
        outputParameterVisitor, network.back())), std::move(currentTarget));
  }

  if (!outputSize)
  {
    outputSize = boost::apply_visitor(outputParameterVisitor,
        network.back()).n_elem;
  }

  return performance;
}

template<typename OutputLayerType, typename InitializationRuleType>
void RNN<OutputLayerType,
         InitializationRuleType
>::Gradient(const arma::mat& parameters,
            const size_t i,
            arma::mat& gradient)
{
  if (gradient.is_empty())
  {
    if (parameter.is_empty())
    {
      ResetParameters();
      reset = true;
    }

    gradient = arma::zeros<arma::mat>(parameter.n_rows, parameter.n_cols);
  }
  else
  {
    gradient.zeros();
  }

  Evaluate(parameters, i, false);

  arma::mat currentGradient = arma::zeros<arma::mat>(parameter.n_rows,
      parameter.n_cols);
  ResetGradients(currentGradient);

  arma::mat input = arma::mat(predictors.colptr(i), predictors.n_rows,
      1, false, true);
  arma::mat target = arma::mat(responses.colptr(i), responses.n_rows,
      1, false, true);

  for (size_t seqNum = 0; seqNum < rho; ++seqNum)
  {
    currentGradient.zeros();

    arma::mat currentTarget = target.rows((rho - seqNum - 1) * targetSize,
        (rho - seqNum) * targetSize - 1);
    currentInput = input.rows((rho - seqNum - 1) * inputSize,
        (rho - seqNum) * inputSize - 1);

    for (size_t l = 0; l < network.size(); ++l)
    {
      boost::apply_visitor(LoadOutputParameterVisitor(
          std::move(moduleOutputParameter)), network[network.size() - 1 - l]);
    }

    if (single && seqNum > 0)
    {
      error.zeros();
    }
    else
    {
      outputLayer.Backward(std::move(boost::apply_visitor(
          outputParameterVisitor, network.back())), std::move(currentTarget),
          std::move(error));
    }

    Backward();
    Gradient();
    gradient += currentGradient;
  }
}

template<typename OutputLayerType, typename InitializationRuleType>
void RNN<OutputLayerType, InitializationRuleType>::ResetParameters()
{
  size_t weights = 0;
  for (LayerTypes& layer : network)
  {
    weights += boost::apply_visitor(weightSizeVisitor, layer);
  }

  parameter.set_size(weights, 1);
  initializeRule.Initialize(parameter, parameter.n_elem, 1);

  size_t offset = 0;
  for (LayerTypes& layer : network)
  {
    offset += boost::apply_visitor(WeightSetVisitor(std::move(parameter),
        offset), layer);

    boost::apply_visitor(resetVisitor, layer);
  }
}

template<typename OutputLayerType, typename InitializationRuleType>
void RNN<OutputLayerType, InitializationRuleType>::ResetDeterministic()
{
  DeterministicSetVisitor deterministicSetVisitor(deterministic);
  std::for_each(network.begin(), network.end(),
      boost::apply_visitor(deterministicSetVisitor));
}

template<typename OutputLayerType, typename InitializationRuleType>
void RNN<OutputLayerType,
         InitializationRuleType
>::ResetGradients(arma::mat& gradient)
{
  size_t offset = 0;
  for (LayerTypes& layer : network)
  {
    offset += boost::apply_visitor(GradientSetVisitor(std::move(gradient),
        offset), layer);
  }
}

template<typename OutputLayerType, typename InitializationRuleType>
void RNN<OutputLayerType,
         InitializationRuleType
>::Forward(arma::mat&& input)
{
  boost::apply_visitor(ForwardVisitor(std::move(input), std::move(
      boost::apply_visitor(outputParameterVisitor, network.front()))),
      network.front());

  for (size_t i = 1; i < network.size(); ++i)
  {
    boost::apply_visitor(ForwardVisitor(
        std::move(boost::apply_visitor(outputParameterVisitor, network[i - 1])),
        std::move(boost::apply_visitor(outputParameterVisitor, network[i]))),
        network[i]);
  }
}

template<typename OutputLayerType, typename InitializationRuleType>
void RNN<OutputLayerType,
         InitializationRuleType
>::Backward()
{
  boost::apply_visitor(BackwardVisitor(
        std::move(boost::apply_visitor(outputParameterVisitor, network.back())),
        std::move(error), std::move(boost::apply_visitor(deltaVisitor,
        network.back()))), network.back());

  for (size_t i = 2; i < network.size(); ++i)
  {
    boost::apply_visitor(BackwardVisitor(
        std::move(boost::apply_visitor(outputParameterVisitor,
        network[network.size() - i])), std::move(boost::apply_visitor(
        deltaVisitor, network[network.size() - i + 1])), std::move(
        boost::apply_visitor(deltaVisitor, network[network.size() - i]))),
        network[network.size() - i]);
  }
}

template<typename OutputLayerType, typename InitializationRuleType>
void RNN<OutputLayerType,
         InitializationRuleType
>::Gradient()
{
  boost::apply_visitor(GradientVisitor(std::move(currentInput), std::move(
      boost::apply_visitor(deltaVisitor, network[1]))), network.front());

  for (size_t i = 1; i < network.size() - 1; ++i)
  {
    boost::apply_visitor(GradientVisitor(
        std::move(boost::apply_visitor(outputParameterVisitor, network[i - 1])),
        std::move(boost::apply_visitor(deltaVisitor, network[i + 1]))),
        network[i]);
  }
}

template<typename OutputLayerType, typename InitializationRuleType>
template<typename Archive>
void RNN<OutputLayerType, InitializationRuleType
>::Serialize(Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(parameter, "parameter");
  ar & data::CreateNVP(rho, "rho");
  ar & data::CreateNVP(single, "single");
  ar & data::CreateNVP(inputSize, "inputSize");
  ar & data::CreateNVP(outputSize, "outputSize");
  ar & data::CreateNVP(targetSize, "targetSize");

  // If we are loading, we need to initialize the weights.
  if (Archive::is_loading::value)
  {
    reset = false;

    size_t offset = 0;
    for (LayerTypes& layer : network)
    {
      offset += boost::apply_visitor(WeightSetVisitor(std::move(parameter),
          offset), layer);

      boost::apply_visitor(resetVisitor, layer);
    }
  }
}

} // namespace ann
} // namespace mlpack

#endif
