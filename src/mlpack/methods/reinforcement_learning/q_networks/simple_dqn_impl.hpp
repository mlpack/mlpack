/**
 * @file simple_dqn_impl.hpp
 * @author Nishant Kumar
 *
 * This file contains the implementation of the simple deep q network.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_SIMPLE_DQN_IMPL_HPP
#define MLPACK_METHODS_RL_SIMPLE_DQN_IMPL_HPP

#include "simple_dqn.hpp"

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>

namespace mlpack {
namespace rl {

using namespace mlpack::ann;

template <typename OutputLayerType, typename InitializationRuleType>
SimpleDQN<OutputLayerType, InitializationRuleType>::SimpleDQN()
{
  // Nothing to do here.
}

template<typename OutputLayerType, typename InitializationRuleType>
SimpleDQN<OutputLayerType, InitializationRuleType>::SimpleDQN(
    const size_t inputDim,
    const size_t h1,
    const size_t h2,
    const size_t outputDim)
{
  network.Add<Linear<>>(inputDim, h1);
  network.Add<ReLULayer<>>();
  network.Add<Linear<>>(h1, h2);
  network.Add<ReLULayer<>>();
  network.Add<Linear<>>(h2, outputDim);
  this->ResetParameters();
}

template<typename OutputLayerType, typename InitializationRuleType>
SimpleDQN<OutputLayerType, InitializationRuleType>::SimpleDQN(
    NetworkType& network) :
    network(std::move(network))
{
  // Nothing to do here.
}

template<typename OutputLayerType, typename InitializationRuleType>
void SimpleDQN<OutputLayerType, InitializationRuleType>::Predict(
    const arma::mat& state, arma::mat& actionValue)
{
  network.Predict(state, actionValue);
}

template<typename OutputLayerType, typename InitializationRuleType>
void SimpleDQN<OutputLayerType, InitializationRuleType>::Forward(
    const arma::mat& state, arma::mat& target)
{
  network.Forward(state, target);
}

template<typename OutputLayerType, typename InitializationRuleType>
void SimpleDQN<OutputLayerType, InitializationRuleType>::Backward(
    const arma::mat& state, const arma::mat& target, arma::mat& gradient)
{
  network.Backward(state, target, gradient);
}

template<typename OutputLayerType, typename InitializationRuleType>
void SimpleDQN<OutputLayerType, InitializationRuleType>::ResetParameters()
{
  network.ResetParameters();
}

} // namespace rl
} // namespace mlpack

#endif
