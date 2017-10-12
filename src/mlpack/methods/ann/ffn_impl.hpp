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

#include "visitor/forward_visitor.hpp"
#include "visitor/backward_visitor.hpp"
#include "visitor/deterministic_set_visitor.hpp"
#include "visitor/gradient_set_visitor.hpp"
#include "visitor/gradient_visitor.hpp"
#include "visitor/set_input_height_visitor.hpp"
#include "visitor/set_input_width_visitor.hpp"

#include <boost/serialization/variant.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {


template<typename OutputLayerType, typename InitializationRuleType>
FFN<OutputLayerType, InitializationRuleType>::FFN(
    OutputLayerType outputLayer, InitializationRuleType initializeRule) :
    outputLayer(std::move(outputLayer)),
    initializeRule(std::move(initializeRule)),
    width(0),
    height(0),
    reset(false),
    numFunctions(0),
    deterministic(true)
{
  /* Nothing to do here */
}

template<typename OutputLayerType, typename InitializationRuleType>
FFN<OutputLayerType, InitializationRuleType>::FFN(
    arma::mat predictors,
    arma::mat responses,
    OutputLayerType outputLayer,
    InitializationRuleType initializeRule) :
    outputLayer(std::move(outputLayer)),
    initializeRule(std::move(initializeRule)),
    width(0),
    height(0),
    reset(false),
    predictors(std::move(predictors)),
    responses(std::move(responses)),
    deterministic(true)
{
  numFunctions = this->responses.n_cols;
}

template<typename OutputLayerType, typename InitializationRuleType>
FFN<OutputLayerType, InitializationRuleType>::~FFN()
{
  std::for_each(network.begin(), network.end(),
      boost::apply_visitor(deleteVisitor));
}

template<typename OutputLayerType, typename InitializationRuleType>
void FFN<OutputLayerType, InitializationRuleType>::ResetData(
    arma::mat predictors, arma::mat responses)
{
  numFunctions = responses.n_cols;
  this->predictors = std::move(predictors);
  this->responses = std::move(responses);
  this->deterministic = true;
  ResetDeterministic();

  if (!reset)
    ResetParameters();
}

template<typename OutputLayerType, typename InitializationRuleType>
template<typename OptimizerType>
void FFN<OutputLayerType, InitializationRuleType>::Train(
      arma::mat predictors,
      arma::mat responses,
      OptimizerType& optimizer)
{
  ResetData(std::move(predictors), std::move(responses));

  // Train the model.
  Timer::Start("ffn_optimization");
  const double out = optimizer.Optimize(*this, parameter);
  Timer::Stop("ffn_optimization");

  Log::Info << "FFN::FFN(): final objective of trained model is " << out
      << "." << std::endl;
}

template<typename OutputLayerType, typename InitializationRuleType>
template<typename OptimizerType>
void FFN<OutputLayerType, InitializationRuleType>::Train(
    arma::mat predictors, arma::mat responses)
{
  numFunctions = responses.n_cols;

  this->predictors = std::move(predictors);
  this->responses = std::move(responses);

  this->deterministic = true;
  ResetDeterministic();

  if (!reset)
    ResetParameters();

  OptimizerType optimizer;

  // Train the model.
  Timer::Start("ffn_optimization");
  const double out = optimizer.Optimize(*this, parameter);
  Timer::Stop("ffn_optimization");

  Log::Info << "FFN::FFN(): final objective of trained model is " << out
      << "." << std::endl;
}

template<typename OutputLayerType, typename InitializationRuleType>
void FFN<OutputLayerType, InitializationRuleType>::Forward(
    arma::mat inputs, arma::mat& results)
{
  if (parameter.is_empty())
    ResetParameters();

  if (!deterministic)
  {
    deterministic = true;
    ResetDeterministic();
  }

  currentInput = std::move(inputs);
  Forward(std::move(currentInput));
  results = boost::apply_visitor(outputParameterVisitor, network.back());
}

template<typename OutputLayerType, typename InitializationRuleType>
double FFN<OutputLayerType, InitializationRuleType>::Backward(
    arma::mat targets, arma::mat& gradients)
{
  double res = outputLayer.Forward(std::move(boost::apply_visitor(
      outputParameterVisitor, network.back())), std::move(targets));

  outputLayer.Backward(std::move(boost::apply_visitor(outputParameterVisitor,
      network.back())), std::move(targets), std::move(error));

  gradients = arma::zeros<arma::mat>(parameter.n_rows, parameter.n_cols);

  Backward();
  ResetGradients(gradients);
  Gradient(std::move(currentInput));

  return res;
}

template<typename OutputLayerType, typename InitializationRuleType>
void FFN<OutputLayerType, InitializationRuleType>::Predict(
    arma::mat predictors, arma::mat& results)
{
  if (parameter.is_empty())
    ResetParameters();

  if (!deterministic)
  {
    deterministic = true;
    ResetDeterministic();
  }

  arma::mat resultsTemp;
  Forward(std::move(arma::mat(predictors.colptr(0),
      predictors.n_rows, 1, false, true)));
  resultsTemp = boost::apply_visitor(outputParameterVisitor,
      network.back()).col(0);

  results = arma::mat(resultsTemp.n_elem, predictors.n_cols);
  results.col(0) = resultsTemp.col(0);

  for (size_t i = 1; i < predictors.n_cols; i++)
  {
    Forward(std::move(arma::mat(predictors.colptr(i),
        predictors.n_rows, 1, false, true)));

    resultsTemp = boost::apply_visitor(outputParameterVisitor,
        network.back());
    results.col(i) = resultsTemp.col(0);
  }
}

template<typename OutputLayerType, typename InitializationRuleType>
double FFN<OutputLayerType, InitializationRuleType>::Evaluate(
    const arma::mat& parameters)
{
  double res = 0;
  for (size_t i = 0; i < predictors.n_cols; ++i)
    res += Evaluate(parameters, i, true);

  return res;
}

template<typename OutputLayerType, typename InitializationRuleType>
double FFN<OutputLayerType, InitializationRuleType>::Evaluate(
    const arma::mat& /* parameters */,
    const size_t begin,
    const size_t batchSize,
    const bool deterministic)
{
  if (parameter.is_empty())
    ResetParameters();

  if (deterministic != this->deterministic)
  {
    this->deterministic = deterministic;
    ResetDeterministic();
  }

  Forward(std::move(predictors.cols(begin, begin + batchSize - 1)));
  double res = outputLayer.Forward(
      std::move(boost::apply_visitor(outputParameterVisitor, network.back())),
      std::move(responses.cols(begin, begin + batchSize - 1)));

  return res;
}

template<typename OutputLayerType, typename InitializationRuleType>
void FFN<OutputLayerType, InitializationRuleType>::Gradient(
    const arma::mat& parameters,
    const size_t begin,
    arma::mat& gradient,
    const size_t batchSize)
{
  if (gradient.is_empty())
  {
    if (parameter.is_empty())
      ResetParameters();

    gradient = arma::zeros<arma::mat>(parameter.n_rows, parameter.n_cols);
  }
  else
  {
    gradient.zeros();
  }

  Evaluate(parameters, begin, batchSize, false);

  outputLayer.Backward(
      std::move(boost::apply_visitor(outputParameterVisitor, network.back())),
      std::move(responses.cols(begin, begin + batchSize - 1)),
      std::move(error));

  Backward();
  ResetGradients(gradient);
  Gradient(std::move(predictors.cols(begin, begin + batchSize - 1)));
}

template<typename OutputLayerType, typename InitializationRuleType>
void FFN<OutputLayerType, InitializationRuleType>::Shuffle()
{
  math::ShuffleData(predictors, responses, predictors, responses);
}

template<typename OutputLayerType, typename InitializationRuleType>
void FFN<OutputLayerType, InitializationRuleType>::ResetParameters()
{
  ResetDeterministic();

  // Reset the network parameter with the given initialization rule.
  NetworkInitialization<InitializationRuleType> networkInit(initializeRule);
  networkInit.Initialize(network, parameter);
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
    reset = true;
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
void FFN<OutputLayerType, InitializationRuleType>::Gradient(arma::mat&& input)
{
  boost::apply_visitor(GradientVisitor(std::move(input), std::move(
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
  ar & data::CreateNVP(currentInput, "currentInput");

  // Be sure to clear other layers before loading.
  if (Archive::is_loading::value)
  {
    std::for_each(network.begin(), network.end(),
        boost::apply_visitor(deleteVisitor));
    network.clear();
  }

  ar & BOOST_SERIALIZATION_NVP(network);

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

    deterministic = true;
    ResetDeterministic();
  }
}

template<typename OutputLayerType, typename InitializationRuleType>
void FFN<OutputLayerType, InitializationRuleType>::Swap(FFN& network)
{
  std::swap(outputLayer, network.outputLayer);
  std::swap(initializeRule, network.initializeRule);
  std::swap(width, network.width);
  std::swap(height, network.height);
  std::swap(reset, network.reset);
  std::swap(this->network, network.network);
  std::swap(predictors, network.predictors);
  std::swap(responses, network.responses);
  std::swap(parameter, network.parameter);
  std::swap(numFunctions, network.numFunctions);
  std::swap(error, network.error);
  std::swap(currentInput, network.currentInput);
  std::swap(deterministic, network.deterministic);
  std::swap(delta, network.delta);
  std::swap(inputParameter, network.inputParameter);
  std::swap(outputParameter, network.outputParameter);
  std::swap(gradient, network.gradient);
};

template<typename OutputLayerType, typename InitializationRuleType>
FFN<OutputLayerType, InitializationRuleType>::FFN(
    const FFN& network):
    outputLayer(network.outputLayer),
    initializeRule(network.initializeRule),
    width(network.width),
    height(network.height),
    reset(network.reset),
    predictors(network.predictors),
    responses(network.responses),
    parameter(network.parameter),
    numFunctions(network.numFunctions),
    error(network.error),
    currentInput(network.currentInput),
    deterministic(network.deterministic),
    delta(network.delta),
    inputParameter(network.inputParameter),
    outputParameter(network.outputParameter),
    gradient(network.gradient)
{
  // Build new layers according to source network
  for (size_t i = 0; i < network.network.size(); ++i)
  {
    this->network.push_back(boost::apply_visitor(copyVisitor,
        network.network[i]));
  }
};

template<typename OutputLayerType, typename InitializationRuleType>
FFN<OutputLayerType, InitializationRuleType>::FFN(
    FFN&& network):
    outputLayer(std::move(network.outputLayer)),
    initializeRule(std::move(network.initializeRule)),
    width(network.width),
    height(network.height),
    reset(network.reset),
    predictors(std::move(network.predictors)),
    responses(std::move(network.responses)),
    parameter(std::move(network.parameter)),
    numFunctions(network.numFunctions),
    error(std::move(network.error)),
    currentInput(std::move(network.currentInput)),
    deterministic(network.deterministic),
    delta(std::move(network.delta)),
    inputParameter(std::move(network.inputParameter)),
    outputParameter(std::move(network.outputParameter)),
    gradient(std::move(network.gradient))
{
  this->network = std::move(network.network);
};

template<typename OutputLayerType, typename InitializationRuleType>
FFN<OutputLayerType, InitializationRuleType>&
FFN<OutputLayerType, InitializationRuleType>::operator = (FFN network)
{
  Swap(network);
  return *this;
};

} // namespace ann
} // namespace mlpack

#endif
