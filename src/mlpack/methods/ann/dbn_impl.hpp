/**
 * @file methods/ann/dbn_impl.hpp
 * @author Himanshu Pathak
 *
 * Definition of the DBN class, which implements feed forward neural networks.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_DBN_IMPL_HPP
#define MLPACK_METHODS_ANN_DBN_IMPL_HPP

// In case it hasn't been included yet.
#include "dbn.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {


template<typename OutputLayerType, typename InitializationRuleType,
         typename... CustomLayers>
DBN<OutputLayerType, InitializationRuleType, CustomLayers...>::DBN(
    OutputLayerType outputLayer,
    InitializationRuleType initializeRule) :
{
  /* Nothing to do here. */
}

template<typename OutputLayerType, typename InitializationRuleType,
         typename... CustomLayers>
DBN<OutputLayerType, InitializationRuleType, CustomLayers...>::~DBN()
{
  std::for_each(network.begin(), network.end(),
      boost::apply_visitor(deleteVisitor));
}

template<typename OutputLayerType, typename InitializationRuleType,
         typename... CustomLayers>
void DBN<OutputLayerType, InitializationRuleType, CustomLayers...>::ResetData(
    arma::mat predictors, arma::mat responses)
{
}

template<typename OutputLayerType, typename InitializationRuleType,
         typename... CustomLayers>
template<typename OptimizerType, typename... CallbackTypes>
double DBN<OutputLayerType, InitializationRuleType, CustomLayers...>::Train(
      arma::mat predictors,
      OptimizerType& optimizer,
      CallbackTypes&&... callbacks)
{
  arma::mat temp = predictors;
  for (size_t i = 0; i < network.size(); ++i)
  {
    OptimizerType opt = optimizer;
    network[i].train(temp, opt);
    arma::mat out;
    network[i].forward(temp, out);
    temp = out;
  }
}

template<typename OutputLayerType, typename InitializationRuleType,
         typename... CustomLayers>
template<typename OptimizerType, typename... CallbackTypes>
double DBN<OutputLayerType, InitializationRuleType, CustomLayers...>::Train(
    arma::mat predictors,
    CallbackTypes&&... callbacks)
{
}

template<typename OutputLayerType, typename InitializationRuleType,
         typename... CustomLayers>
template<typename PredictorsType, typename ResponsesType>
void DBN<OutputLayerType, InitializationRuleType, CustomLayers...>::Forward(
    const PredictorsType& inputs, ResponsesType& results)
{
}

template<typename OutputLayerType, typename InitializationRuleType,
         typename... CustomLayers>
void DBN<OutputLayerType, InitializationRuleType, CustomLayers...>::Predict(
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
  Forward(arma::mat(predictors.colptr(0), predictors.n_rows, 1, false, true));
  resultsTemp = boost::apply_visitor(outputParameterVisitor,
      network.back()).col(0);

  results = arma::mat(resultsTemp.n_elem, predictors.n_cols);
  results.col(0) = resultsTemp.col(0);

  for (size_t i = 1; i < predictors.n_cols; ++i)
  {
    Forward(arma::mat(predictors.colptr(i), predictors.n_rows, 1, false, true));

    resultsTemp = boost::apply_visitor(outputParameterVisitor,
        network.back());
    results.col(i) = resultsTemp.col(0);
  }
}

template<typename OutputLayerType, typename InitializationRuleType,
         typename... CustomLayers>
void DBN<OutputLayerType, InitializationRuleType, CustomLayers...>::Shuffle()
{
  math::ShuffleData(predictors, responses, predictors, responses);
}

template<typename OutputLayerType, typename InitializationRuleType,
         typename... CustomLayers>
template<typename Archive>
void DBN<OutputLayerType, InitializationRuleType, CustomLayers...>::serialize(
    Archive& ar, const unsigned int version)
{
}

template<typename OutputLayerType, typename InitializationRuleType,
         typename... CustomLayers>
void DBN<OutputLayerType, InitializationRuleType,
         CustomLayers...>::Swap(DBN& network)
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
  std::swap(deterministic, network.deterministic);
  std::swap(delta, network.delta);
  std::swap(inputParameter, network.inputParameter);
  std::swap(outputParameter, network.outputParameter);
  std::swap(gradient, network.gradient);
};

template<typename OutputLayerType, typename InitializationRuleType,
         typename... CustomLayers>
DBN<OutputLayerType, InitializationRuleType, CustomLayers...>::DBN(
    const DBN& network):
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

template<typename OutputLayerType, typename InitializationRuleType,
         typename... CustomLayers>
DBN<OutputLayerType, InitializationRuleType, CustomLayers...>::DBN(
    DBN&& network):
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
    deterministic(network.deterministic),
    delta(std::move(network.delta)),
    inputParameter(std::move(network.inputParameter)),
    outputParameter(std::move(network.outputParameter)),
    gradient(std::move(network.gradient))
{
  this->network = std::move(network.network);
};

template<typename OutputLayerType, typename InitializationRuleType,
         typename... CustomLayers>
DBN<OutputLayerType, InitializationRuleType, CustomLayers...>&
DBN<OutputLayerType, InitializationRuleType,
    CustomLayers...>::operator = (DBN network)
{
  Swap(network);
  return *this;
};

} // namespace ann
} // namespace mlpack

#endif
