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


template<typename OutputType, typename InitializationRuleType>
DBN<OutputType, InitializationRuleType>::DBN(arma::mat predictors) :
  predictors(std::move(predictors))
{
  /* Nothing to do here. */
}

template<typename OutputType, typename InitializationRuleType>
void DBN<OutputType, InitializationRuleType>::ResetData(
    arma::mat predictors, arma::mat responses)
{
}

template<typename OutputType, typename InitializationRuleType>
template<typename OptimizerType, typename... CallbackTypes>
double DBN<OutputType, InitializationRuleType>::Train(
      OptimizerType& optimizer,
      CallbackTypes&&... callbacks)
{
  double var;
  arma::mat temp = predictors;
  for (size_t i = 0; i < network.size(); ++i)
  {
    OptimizerType opt = optimizer;
    var = network[i].Train(temp, opt);
    arma::mat out;
    network[i].Forward(temp, out);
    temp = out;
  }
  return var;
}

template<typename OutputType, typename InitializationRuleType>
template<typename OptimizerType, typename... CallbackTypes>
double DBN<OutputType, InitializationRuleType>::Train(
    const double layerNumber,
    OptimizerType& optimizer,
    CallbackTypes&&... callbacks)
{
  double var;
  if(layerNumber > network.size())
  {
    Log::Warn(" Entered value is greater than numder of layers");
  }
  arma::mat temp = predictors;
  for (size_t i = 0; i < layerNumber - 1; ++i)
  {
    arma::mat out;
    network[i].Forward(temp, out);
    temp = out;
  }
  var = network[layerNumber].Train(temp, optimizer);
  return var;

}

template<typename OutputType, typename InitializationRuleType>
template<typename PredictorsType, typename ResponsesType>
void DBN<OutputType, InitializationRuleType>::Forward(
    const PredictorsType& inputs, ResponsesType& results)
{
  arma::mat temp = inputs;
  for (size_t i = 0; i < network.size(); ++i)
  {
    arma::mat out;
    network[i].Forward(temp, out);
    temp = out;
  }
  results = temp;
}

template<typename OutputType, typename InitializationRuleType>
void DBN<OutputType, InitializationRuleType>::Shuffle()
{
  predictors = predictors.cols(arma::shuffle(arma::linspace<arma::uvec>(0,
      predictors.n_cols - 1, predictors.n_cols)));
}

template<typename OutputType, typename InitializationRuleType>
template<typename Archive>
void DBN<OutputType, InitializationRuleType>::serialize(
    Archive& ar, const unsigned int version)
{
}

} // namespace ann
} // namespace mlpack

#endif
