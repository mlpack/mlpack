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


template<typename OutputLayerType, typename InitializationRuleType>
DBN<OutputLayerType, InitializationRuleType>::DBN(
    OutputLayerType outputLayer,
    InitializationRuleType initializeRule) :
{
  /* Nothing to do here. */
}

template<typename OutputLayerType, typename InitializationRuleType>
DBN<OutputLayerType, InitializationRuleType>::~DBN()
{
  std::for_each(network.begin(), network.end(),
      boost::apply_visitor(deleteVisitor));
}

template<typename OutputLayerType, typename InitializationRuleType>
void DBN<OutputLayerType, InitializationRuleType>::ResetData(
    arma::mat predictors, arma::mat responses)
{
}

template<typename OutputLayerType, typename InitializationRuleType>
template<typename OptimizerType, typename... CallbackTypes>
double DBN<OutputLayerType, InitializationRuleType>::Train(
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

template<typename OutputLayerType, typename InitializationRuleType>
template<typename OptimizerType, typename... CallbackTypes>
double DBN<OutputLayerType, InitializationRuleType>::Train(
    arma::mat predictors,
    CallbackTypes&&... callbacks)
{
}

template<typename OutputLayerType, typename InitializationRuleType>
template<typename PredictorsType, typename ResponsesType>
void DBN<OutputLayerType, InitializationRuleType>::Forward(
    const PredictorsType& inputs, ResponsesType& results)
{
}

template<typename OutputLayerType, typename InitializationRuleType>
void DBN<OutputLayerType, InitializationRuleType>::Shuffle()
{
  math::ShuffleData(predictors, responses, predictors, responses);
}

template<typename OutputLayerType, typename InitializationRuleType>
template<typename Archive>
void DBN<OutputLayerType, InitializationRuleType>::serialize(
    Archive& ar, const unsigned int version)
{
}

} // namespace ann
} // namespace mlpack

#endif
