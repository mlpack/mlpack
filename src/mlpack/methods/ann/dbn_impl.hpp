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


template<typename InputType,
         typename OutputType,
         typename InitializationRuleType>
DBN<InputType, OutputType, InitializationRuleType>::DBN(InputType predictors) :
  predictors(std::move(predictors))
{
  /* Nothing to do here. */
}

template<typename InputType,
         typename OutputType,
         typename InitializationRuleType>
void DBN<InputType, OutputType, InitializationRuleType>::Shuffle()
{
  predictors = predictors.cols(arma::shuffle(arma::linspace<arma::uvec>(0,
      predictors.n_cols - 1, predictors.n_cols)));
}

template<typename InputType,
         typename OutputType,
         typename InitializationRuleType>
void DBN<InputType, OutputType, InitializationRuleType>::Reset()
{
  // Reset all the RBM present in the network.
  for (size_t i = 0; i < network.size(); ++i)
  {
    network[i].Reset();
  }
}

template<typename InputType,
         typename OutputType,
         typename InitializationRuleType>
void DBN<InputType, OutputType, InitializationRuleType>::SetBias()
{
  // Setting hidden and visible bias of all RBM layers.
  for (size_t i = 0; i < network.size(); ++i)
  {
    network[i].VisibleBias().ones();
    network[i].HiddenBias().ones();
  }
}

template<typename InputType,
         typename OutputType,
         typename InitializationRuleType>
template<typename OptimizerType, typename... CallbackTypes>
double DBN<InputType, OutputType, InitializationRuleType>::Train(
      OptimizerType& optimizer,
      CallbackTypes&&... callbacks)
{
  double var;
  arma::mat temp = predictors;
  // Training all the layers by greedy approach.
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

template<typename InputType,
         typename OutputType,
         typename InitializationRuleType>
template<typename OptimizerType, typename... CallbackTypes>
double DBN<InputType, OutputType, InitializationRuleType>::Train(
    const double layerNumber,
    OptimizerType& optimizer,
    CallbackTypes&&... callbacks)
{
  double var;
  if (layerNumber > network.size())
  {
    Log::Warn << " LayerNumber is greater than network size";
    return 0;
  }
  arma::mat temp = predictors;
  // Getting output from previous layers to train a single layer.
  for (size_t i = 0; i < layerNumber - 1; ++i)
  {
    arma::mat out;
    network[i].Forward(temp, out);
    temp = out;
  }
  // Training a single layer.
  var = network[layerNumber].Train(temp, optimizer);
  return var;
}

template<typename InputType,
         typename OutputType,
         typename InitializationRuleType>
void DBN<InputType, OutputType, InitializationRuleType>::Forward(
    const InputType& inputs, OutputType& results)
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

template<typename InputType,
         typename OutputType,
         typename InitializationRuleType>
template<typename Archive>
void DBN<InputType, OutputType, InitializationRuleType>::serialize(
    Archive& ar, const unsigned int version)
{
  ar & BOOST_SERIALIZATION_NVP(predictors);
  ar & BOOST_SERIALIZATION_NVP(results);
  ar & BOOST_SERIALIZATION_NVP(network);
}

} // namespace ann
} // namespace mlpack

#endif
